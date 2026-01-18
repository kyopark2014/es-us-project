import utils
import langgraph_agent
import mcp_config
import chat
import langgraph_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger("qa-agent")

async def run_qa_agent(query, containers):
    global index
    index = 0

    image_url = []
    references = []

    mcp_servers = ["knowledge base", "tavily-search"]

    mcp_json = mcp_config.load_selected_config(mcp_servers)
    logger.info(f"mcp_json: {mcp_json}")

    server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)
    logger.info(f"server_params: {server_params}")    

    try:
        client = MultiServerMCPClient(server_params)
        logger.info(f"MCP client created successfully")
        
        tools = await client.get_tools()
        logger.info(f"get_tools() returned: {tools}")
        
        if tools is None:
            logger.error("tools is None - MCP client failed to get tools")
            tools = []
        
        tool_list = [tool.name for tool in tools] if tools else []
        logger.info(f"tool_list: {tool_list}")
        
    except Exception as e:
        logger.error(f"Error creating MCP client or getting tools: {e}")
        return "MCP 설정을 확인하세요."

    system_prompt = (
        "당신은 숙련된 QA 엔지니어입니다."
        "다음의 주어진 주제를 참조하여 retrieve tool을 이용해 필요한 정보를 수집합니다."
        "수집된 정보를 이용하여 Test Case를 생성합니다."
        "Test Case에는 선행조건, 테스트단계, 예상결과에 대해 세부적인 내용을 제시합니다."
        "Test Case는 중복되지 않도록 작성해주세요."
    )
    
    app = langgraph_agent.buildChatAgent(tools)
    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": "qa-agent"},
        "tools": tools,
        "system_prompt": system_prompt
    }        
    
    inputs = {
        "messages": [HumanMessage(content=query)]
    }
            
    result = ""
    tool_used = False  # Track if tool was used
    tool_name = toolUseId = ""
    async for output in app.astream(inputs, config, stream_mode="messages"):
        # logger.info(f"output: {output}")

        # Handle tuple output (message, metadata)
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], AIMessageChunk):
            message = output[0]    
            input = {}        
            if isinstance(message.content, list):
                for content_item in message.content:
                    if isinstance(content_item, dict):
                        if content_item.get('type') == 'text':
                            text_content = content_item.get('text', '')
                            # logger.info(f"text_content: {text_content}")
                            
                            # If tool was used, start fresh result
                            if tool_used:
                                result = text_content
                                tool_used = False
                            else:
                                result += text_content
                                
                            # logger.info(f"result: {result}")                
                            chat.update_streaming_result(containers, result, "markdown")

                        elif content_item.get('type') == 'tool_use':
                            logger.info(f"content_item: {content_item}")      
                            if 'id' in content_item and 'name' in content_item:
                                toolUseId = content_item.get('id', '')
                                tool_name = content_item.get('name', '')
                                logger.info(f"tool_name: {tool_name}, toolUseId: {toolUseId}")

                                chat.tool_info_list[toolUseId] = index                     
                                chat.tool_name_list[toolUseId] = tool_name     
                                                                        
                            if 'partial_json' in content_item:
                                partial_json = content_item.get('partial_json', '')
                                logger.info(f"partial_json: {partial_json}")
                                
                                if toolUseId not in chat.tool_input_list:
                                    chat.tool_input_list[toolUseId] = ""                                
                                chat.tool_input_list[toolUseId] += partial_json
                                input = chat.tool_input_list[toolUseId]
                                logger.info(f"input: {input}")

                                logger.info(f"tool_name: {tool_name}, input: {input}, toolUseId: {toolUseId}")
                                # add_notification(containers, f"Tool: {tool_name}, Input: {input}")
                                index = chat.tool_info_list[toolUseId]

                                if chat.debug_mode == "Enable":
                                    containers['notification'][index].info(f"Tool: {tool_name}, Input: {input}")
                        
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], ToolMessage):
            message = output[0]
            logger.info(f"ToolMessage: {message.name}, {message.content}")
            tool_name = message.name
            toolResult = message.content
            toolUseId = message.tool_call_id
            logger.info(f"toolResult: {toolResult}, toolUseId: {toolUseId}")
            
            if chat.debug_mode == "Enable":
                chat.add_notification(containers, f"Tool Result: {toolResult}")
            
            tool_used = True
            
            content, urls, refs = chat.get_tool_info(tool_name, toolResult)
            if refs:
                for r in refs:
                    references.append(r)
                logger.info(f"refs: {refs}")
            if urls:
                for url in urls:
                    image_url.append(url)
                logger.info(f"urls: {urls}")

            if content:
                logger.info(f"content: {content}")        
    
    if not result:
        result = "답변을 찾지 못하였습니다."        
    logger.info(f"result: {result}")

    if references:
        ref = "\n\n### Reference\n"
        for i, reference in enumerate(references):
            page_content = reference['content'][:100].replace("\n", "")
            ref += f"{i+1}. [{reference['title']}]({reference['url']}), {page_content}...\n"    
        result += ref
    
    if containers is not None:
        containers['notification'][index].markdown(result)
    
    return result
