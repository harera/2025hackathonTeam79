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
class CosmosDBClient:
    """封装与Cosmos DB的交互（使用Connection String连接）"""
    
    def __init__(self):
        self.logger = logging.getLogger("cosmos_client")
        # 从环境变量获取连接字符串
        connection_string = os.getenv("COSMOS_CONNECTION_STRING")
        if not connection_string:
            error_msg = "COSMOS_CONNECTION_STRING environment variable is not set"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            self.client = CosmosClient.from_connection_string(connection_string)
            self.logger.info("Successfully connected to Cosmos DB")
        except Exception as e:
            error_msg = f"Failed to connect to Cosmos DB: {str(e)}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """根据user_id获取用户信息"""
        try:
            self.logger.info(f"Fetching user info for ID: {user_id}")
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
                )
            ]
            
            if not items:
                error_msg = f"User not found with ID: {user_id}"
                self.logger.warning(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"Successfully retrieved user info for ID: {user_id}")
            return items[0]
        except Exception as e:
            error_msg = f"Failed to query Cosmos DB: {str(e)}"
            self.logger.error(error_msg)
            raise
        finally:
            try:
                await self.client.close()
                self.logger.info("Cosmos DB connection closed")
            except Exception as e:
                self.logger.error(f"Error closing Cosmos DB connection: {str(e)}")


# --- Foundry 客户端 ---
class FoundryIncomeAgent:
    """封装与Foundry AI的交互"""
    
    def __init__(self):
        self.client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str="eastus2.api.azureml.ms;0dd70ca3-209c-42b7-8dbe-2de878b7b127;odl-sandbox-1660037-02;aironwomen",
        )  
    
    async def get_EvidenceOfFraud_blob_path(self, userName: str) -> str:
        """
        调用Foundry Agent获取调用web api获取用户欺诈风险调查结果
        """    
        try:
            agent_definition = await self.client.agents.get_agent(agent_id="asst_zvCohHloGovi4OvDnQLTKkbd")
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
        self.logger = logging.getLogger("fraud_evaluator")
        
        self.kernel = sk.Kernel()
        self.logger.info("初始化 SK Kernel")
        
        self.kernel.add_service(
            AzureChatCompletion(
                service_id="credit_ai",
                deployment_name="gpt-4o-mini",
                endpoint="https://hackathonfound6454649233.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                api_key="U4fD1XnytGhxvUlsRYSRL472aImrcK17LbNS5CQfGOTRpkWKAwpzJQQJ99BDACHYHv6XJ3w3AAAAACOGIngS",
                api_version="2024-12-01-preview"
            )
        )
        self.logger.info("添加 AzureChatCompletion 服务到 Kernel")
        
        # 初始化真正的Agent实例
        self.risk_agent = ChatCompletionAgent(
            kernel=self.kernel,
            name="risk_agent",
            instructions="""
            You are an expert in anti fraud in banks and will pay special attention to the risk points of information：
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
        self.logger.info("初始化风险分析 Agent")
        
        self.service_agent = ChatCompletionAgent(
            kernel=self.kernel,
            name="service_agent",
            instructions="""
            You are the customer manager of the bank, responsible for helping users explain the rationality of materials. You will focus on the user's contributions and advantages, assist users in completing loan approval, and list the following key information of the user:
            1. Special and reasonable circumstances
            2. Possibility of supplementary materials
            3. Customer historical contributions
            """,
            description="customer manager"
        )
        self.logger.info("初始化客户服务 Agent")
        
        # 群聊设置
        self.group_chat = AgentGroupChat(
            agents=[self.risk_agent, self.service_agent],
            selection_strategy=SequentialSelectionStrategy(),
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=3)
        )
        self.logger.info("初始化 Agent 群聊")
    
    async def evaluate(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info("开始欺诈评估")
            # 1. 直接将证据数据转换为字符串作为初始提示
            evidence_str = json.dumps(evidence_data, ensure_ascii=False)
            self.logger.info(f"准备评估数据: {evidence_str}")
            
            # 2. 创建初始消息内容
            initial_message = ChatMessageContent(
                role="user",
                content=f"请分析以下欺诈证据数据：\n{evidence_str}"
            )
            
            # 3. 直接添加到群聊（不需要ChatHistory）
            self.logger.info("开始 Agent 群聊")
            await self.group_chat.add_chat_message(initial_message)
            
            # 4. 收集响应
            discussion = []
            async for msg in self.group_chat.invoke():
                if isinstance(msg, ChatMessageContent):
                    discussion.append(f"{msg.role}: {msg.content}")
                    self.logger.info(f"收到 Agent 消息: {msg.role}: {msg.content}")

            # 5. 生成最终决策
            if not discussion:
                self.logger.error("讨论记录为空，无法生成决策")
                raise ValueError("讨论记录为空，无法生成决策")
            
            # 1. 将讨论内容格式化为更清晰的对话记录
            formatted_discussion = "\n".join([
                f"【{msg.split(':')[0]}】{msg.split(':', 1)[1].strip()}"
                for msg in discussion
            ])
        
            self.logger.info("=== 完整讨论记录 ===")
            self.logger.info(formatted_discussion)

            # 2. 创建更明确的总结提示
            summary_prompt = f"""
            Please make a final decision based on the following expert discussions, adopt more positive opinions, and try to approve the loan as much as possible:
        
            {formatted_discussion}
        
            Decision requirements:
            1. Comprehensive risk analysis and service recommendations
            2. Clear conclusion: approved or rejected
            3. Concise reasons (2-3 sentences)
        
            Please return strictly in the following format:
            Decision: [APPROVED/REJECTED]
            Reason: [your reasons]
            """
            
            self.logger.info("调用 SK 函数生成最终决策")
            final_decision = await self.kernel.invoke(
                self.kernel.add_function(
                    plugin_name="DecisionServices",
                    function_name="MakeFinalDecision",
                    prompt_template_config=PromptTemplateConfig(
                        template=summary_prompt,
                        function_name="make_final_decision"
                    )
                ),
                arguments=KernelArguments()
            )
            
            self.logger.info(f"最终决策结果: {final_decision}")
            
            return {
                "status": "success",
                "decision": str(final_decision),
                "discussion": formatted_discussion,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"SK 函数调用失败: {str(e)}")
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
        print(f"获取到用户信息: {json.dumps(user_info, indent=2)}")

        investigation_result = await foundry_agent.get_EvidenceOfFraud_blob_path(user_name)
        print(f"获取到欺诈调查结果: {json.dumps(investigation_result, indent=2)}")

        # 评估欺诈风险
        result = await evaluator.evaluate(investigation_result)
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

async def test_fraud_evaluation():
    """测试欺诈评估功能"""
    try:
        # 使用测试 user_id
        user_id = "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca"  # 测试用的用户ID
        
        # 执行欺诈分析工作流
        result = await fraud_analysis_workflow(user_id)
        
        # 打印结果
        print("\n=== 欺诈评估结果 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_fraud_evaluation())
