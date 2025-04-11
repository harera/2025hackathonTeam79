import os
import json
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
import logging

# Azure 服务SDK
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient
from azure.cosmos.aio import CosmosClient
from azure.ai.projects.aio import AIProjectClient
from azure.core.credentials import AzureKeyCredential

# Semantic Kernel
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat, AzureAIAgent
from semantic_kernel.agents.strategies import SequentialSelectionStrategy, DefaultTerminationStrategy
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

# --- Cosmos DB 客户端 ---
# --- Cosmos DB 客户端 ---
class CosmosDBClient:
    """封装与Cosmos DB的交互（使用Connection String连接）"""
    
    def __init__(self):
        self.client = CosmosClient.from_connection_string(
            "AccountEndpoint=https://=;"
        )
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """根据user_id获取用户信息"""
        try:
            database = self.client.get_database_client("cosmicworks")
            container = database.get_container_client("userinfo")
            
            # 使用参数化查询（更安全）
            query = "SELECT * FROM c WHERE c.id = @user_id"
            parameters = [
                {"name": "@user_id", "value": user_id}
            ]
            
            items = [
                item 
                async for item in container.query_items(
                    query=query,
                    parameters=parameters
                    # 不再需要 enable_cross_partition_query
                )
            ]
            
            if not items:
                raise ValueError(f"未找到用户ID: {user_id}")
            
            return items[0]
        except Exception as e:
            print(f"[Cosmos DB] 查询失败: {str(e)}")
            raise
        finally:
            await self.client.close()


# --- Foundry 客户端 ---
class FoundryIncomeAgent:
    """封装与Foundry AI的交互"""
    
    def __init__(self):
        self.client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str="",
        )  
    
    async def get_EvidenceOfFraud_blob_path(self, userName: str) -> str:
        """
        调用Foundry Agent获取调用web api获取用户欺诈风险调查结果
        """    
        try:
            agent_definition = await self.client.agents.get_agent(agent_id="")
            getBlobPath_agent = AzureAIAgent(client=self.client, definition=agent_definition)
            response = await getBlobPath_agent.get_response(messages=userName)
            
            # 安全处理响应
            if hasattr(response, 'content'):
                return str(response.content)
            elif isinstance(response, str):
                return response
            else:
                raise ValueError("无法解析Foundry Agent响应")
        except Exception as e:
            print(f"[Foundry] 调用失败: {str(e)}")
            raise


# --- 欺诈检测核心 ---
class FraudEvaluator:
    def __init__(self):
        
        self.kernel = sk.Kernel()
        self.kernel.add_service(
            AzureChatCompletion(
                service_id="credit_ai",
                deployment_name="gpt-4o-mini",
                endpoint="https://hackathonfound6454649233.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                api_key="U",
                api_version="2024-12-01-preview"
            )
        )
        # 初始化真正的Agent实例
        self.risk_agent = ChatCompletionAgent(
            kernel=self.kernel,
            name="risk_agent",
            instructions="""
            You are an expert in anti fraud in banks and will pay special attention to the risk points of information：
            Before each speak,  show 【Challenger Agent】.
            Start by making your point: you claim that the transaction is illegal and that there is a risk of fraud.
            In a sentence or two, explain the following reasons:
            1.Identity Consistency:Name, ID number, and phone number match official records.No discrepancies in personal information.
            2.Social Insurance Verification:Consistent payments for the last 3 months from the same employer.Employer name matches the application details .
            3.Loan repayment:No overdue payments in the last 3 months.Full repayment  and on-time status.
            4.Is the remaining principal sufficient after repayment
            5.Income Stability
            6.Legal & Credit History:No criminal records.Zero overdue records in credit history .
            
            Risk points must be clearly identified upon discovery.To completely eliminate the occurrence of fraud.
            """,
            description="anti fraud"
        )
        
        self.service_agent = ChatCompletionAgent(
            kernel=self.kernel,
            name="service_agent",
            instructions="""

            You are the customer manager of the bank, responsible for helping users explain the rationality of materials. You will focus on the user's contributions and advantages, assist users in completing loan approval, and list the following key information of the user:
            Before each speak, show 【Defender Agent】.
            Start by stating your point of view: you assert the reasonableness of the transaction and that there is no fraud.
            In a sentence or two, explain the following reasons:
            1. Demonstrate the reasonableness of the special circumstances
            2. Possibility of additional materials
            3. Customer Historical Contribution
            """,
            description="customer manager"
        )
        
        
        # 群聊设置
        self.group_chat = AgentGroupChat(
            agents=[self.risk_agent, self.service_agent],
            selection_strategy=SequentialSelectionStrategy(),
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=4)
        )
        
       
    
    async def evaluate(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 1. 直接将证据数据转换为字符串作为初始提示
            evidence_str = json.dumps(evidence_data, ensure_ascii=False)
            
            # 2. 创建初始消息内容
            initial_message = ChatMessageContent(
                role="user",
                content=f"Analyze the following evidence of fraud data:\n{evidence_str}"
            )
            
            # 3. 直接添加到群聊（不需要ChatHistory）
            await self.group_chat.add_chat_message(initial_message)
            
            # 4. 收集响应
            discussion = []
            async for msg in self.group_chat.invoke():
                if isinstance(msg, ChatMessageContent):
                    discussion.append(f"{msg.role}: {msg.content}")


            # 5. 生成最终决策
            if not discussion:
                raise ValueError("讨论记录为空，无法生成决策")
            
            # 1. 将讨论内容格式化为更清晰的对话记录
            formatted_discussion = "\n".join([
                f"{msg.split(':', 1)[1].strip()}" #【{msg.split(':')[0]}】
                for msg in discussion
            ])
        
            print("=== 完整讨论记录 ===")
            print(formatted_discussion)

            # 2. 创建更明确的总结提示
            summary_prompt = f"""
            Please make a final decision based on the following expert discussions, adopt more positive opinions, and try to approve the loan as much as possible:
        
            {formatted_discussion}
        
            Decision requirements:
            1. Clear conclusion: approved or rejected
            2. Comprehensive risk analysis and service recommendations
            3. Concise reasons (2-3 sentences)
            Before each speak, show 【Arbiter Agent】
            You first say whether there is a risk of fraud in the conclusion, and then briefly explain the reason (1~2 sentences)
            
            """

            # 3. 直接使用AzureChatCompletion进行总结
            chat_service = self.kernel.get_service("credit_ai")
            chat_history = ChatHistory()
            chat_history.add_user_message(summary_prompt)
            settings =PromptExecutionSettings(
                service_id="credit_ai",
                temperature=0.7, 
                max_tokens=500
                )
            # 4. 调用并严格验证结果
            result = await chat_service.get_chat_message_contents(
                chat_history=chat_history,
                kernel=self.kernel,
                settings=settings)
            decision_content = str(result[0].content)

            return decision_content

        except Exception as e:
            print(f"评估过程出错: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

async def fraud_analysis_workflow(user_id: str) -> Dict[str, Any]:
    """欺诈分析工作流"""
    try:
        # 初始化客户端
        foundry_agent = FoundryIncomeAgent()
        evaluator = FraudEvaluator()
        cosmos_client = CosmosDBClient()
        

        # 获取用户名
        user_info = await cosmos_client.get_user_info(user_id)
        user_name = user_info.get("name")
        if not user_name:
            raise ValueError("用户信息中缺少姓名")
        print(f"get user info: {json.dumps(user_info, indent=2)}")
        user_name_str = "tell me about fraud assessment information:" + user_name
        print(user_name_str)
        investigation_result = await foundry_agent.get_EvidenceOfFraud_blob_path(user_name_str)
        investigation = {json.dumps(investigation_result, ensure_ascii=False, indent=2)}
        print(f"Obtain anti-fraud data collection information: {json.dumps(investigation_result, ensure_ascii=False, indent=2)}")

        # 评估欺诈风险
        result = await evaluator.evaluate(str(investigation))
        return result
        
        # 加载数据
        # evidence_data = await blob_loader.load_income_data("test.json")
        # print(f"加载到证据数据: {json.dumps(evidence_data, indent=2)}")
    except Exception as e:
        return {
            "status": "error",
            "message": f"工作流执行失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
if __name__ == "__main__":
    async def main():
        test_filename = "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca"
        try:
            result = await fraud_analysis_workflow(test_filename)
            print("\n最终结果:")
            print(result)
        except Exception as e:
            print(f"主程序错误: {str(e)}")

    asyncio.run(main())