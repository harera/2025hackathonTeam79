"""
Azure AI 信用评分系统
功能：
1. 根据用户输入文件名，调用Foundry Agent获取Blob存储路径
2. 从Azure Blob Storage加载收入证明JSON
3. 使用Azure OpenAI分析信用风险
4. 计算FICO信用评分（300-850）
环境要求：
pip install semantic-kernel>=1.0.0 azure-ai-foundry azure-storage-blob python-dotenv
"""

import os
import json
import asyncio
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import logging

# Azure 服务SDK
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient

from azure.ai.projects.aio import AIProjectClient
from azure.core.credentials import AzureKeyCredential

# Semantic Kernel
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

from autogen import ConversableAgent, register_function

# --- Foundry 客户端 ---
class FoundryIncomeAgent:
    """封装与Foundry AI的交互"""
    
    def __init__(self):
        self.client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str="eastus2.api.azureml.ms;0dd70ca3-209c-42b7-8dbe-2de878b7b127;odl-sandbox-1660037-02;aironwomen",
        )  
        print(self.client)
    async def get_income_blob_path(self, filename: str) -> str:
        """
        调用Foundry Agent获取收入证明的Blob路径
        """    
        try:
            agent_definition = await self.client.agents.get_agent(agent_id="asst_cb5vNNrQzB8BDMGms51SMt7A")
            getBlobPath_agent = AzureAIAgent(client=self.client, definition=agent_definition)
            thread: AzureAIAgentThread = None
            response = await getBlobPath_agent.get_response(messages=filename, thread=thread)
            res = response.message.items[0].text
            print(f"# {response.name}: {response}")
            print(response.message.items[0].text)
            print(f"Full response: {vars(response)}")  # 查看所有属性
            print(f"Content type: {type(response)}")  # 确认类型
            thread = response.thread
            
            if not response:
                raise ValueError("Foundry Agent未返回有效的blob_path")
                
            return response.message.items[0].text
            
        except Exception as e:
            print(f"[Foundry] 调用失败: {str(e)}")
            raise

# --- Azure Blob 存储操作 ---
class IncomeProofLoader:
    """从Blob Storage加载收入证明"""
    
    def __init__(self):
        # 从环境变量获取容器名称
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME_JSON", "ocrjson")
        self.logger = logging.getLogger("income_loader")
        self.logger.info(f"使用容器名称: {container_name}")
        
        self.client = BlobServiceClient.from_connection_string(
            "DefaultEndpointsProtocol=https;AccountName=hackathonfound1559274341;AccountKey=Foer+fIx7QM9ihnWxnwJrnRx/GA3YydT3UJ4vzVwr4xkfAdHoUacaOqx5CgVwt07vrgLl1N2IA3l+AStAfISnA==;EndpointSuffix=core.windows.net"
        )
        self.container_name = container_name
    
    async def load_income_data(self, blob_path: str) -> Dict[str, Any]:
        """异步加载JSON格式的收入证明"""
        try:
            self.logger.info(f"尝试从容器 {self.container_name} 加载blob: {blob_path}")
            blob_client = self.client.get_blob_client(
                container=self.container_name,
                blob=blob_path
            )
            
            # 下载blob内容
            downloader = await blob_client.download_blob()
            data = await downloader.readall()
            
            # 解析JSON数据
            result = json.loads(data.decode('utf-8'))
            self.logger.info(f"成功加载blob数据: {blob_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"[Blob Storage] 加载失败: {blob_path}, 错误: {str(e)}")
            raise
    
# --- 信用评估核心 ---
class CreditEvaluator:
    """极简信用评估（仅基于月收入）"""
    
    def __init__(self):
        self.logger = logging.getLogger("credit_evaluator")
        self.logger.info("初始化 SK Kernel")
        
        self.kernel = sk.Kernel()
        self.logger.info("SK Kernel 创建成功")
        
        self.kernel.add_service(
            AzureChatCompletion(
                service_id="credit_ai",
                deployment_name="gpt-4o-mini",
                endpoint="https://hackathonfound6454649233.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                api_key="U4fD1XnytGhxvUlsRYSRL472aImrcK17LbNS5CQfGOTRpkWKAwpzJQQJ99BDACHYHv6XJ3w3AAAAACOGIngS",
                api_version="2024-12-01-preview"
            )
        )
        self.logger.info("AzureChatCompletion 服务已添加到 Kernel")
        
        self.assess_credit = self.kernel.add_function(
            plugin_name="CreditServices",
            function_name="AssessCreditRisk",
            prompt_template_config=PromptTemplateConfig(
            template="""
            You are a professional credit risk analyst.
            You now have the extracted employment certificate information: 
            {{$COE}}
            bank statements information: 
            {{$BS}}
            You need to analyze these two data sources together to identify potential risk signals:
            1.Check if the salary level in the employment certificate matches the bank statements - is there any income exaggeration?
            2.If the applicant has debt, calculate the DTI (including mortgage, auto loans, and credit card payments) and verify if it exceeds the 50% threshold.
            3.Use your professional judgment to determine if the user's credit qualifies for loan services.
            Credit rating standards:
            A: All data consistent, no risks
            B: Minor discrepancies, manual review advised
            C: High-risk (e.g., DTI exceeds limit or document falsification)
            Output format only:
            The user's credit grade is [A/B/C]. [1-2 sentence risk summary]
            """,
            function_name="assess_credit"
            )
        )
        self.logger.info("信用评估函数已添加到 Kernel")
    
    async def evaluate(self, CertificateOfEmployment: Dict[str, Any],BankStatements: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info("开始信用评估流程")
            self.logger.debug(f"输入数据 - CertificateOfEmployment: {CertificateOfEmployment}")
            self.logger.debug(f"输入数据 - BankStatements: {BankStatements}")
            
            # 检查输入数据是否为空
            if not CertificateOfEmployment or not BankStatements:
                raise ValueError("输入数据为空")
            
            # 获取分析结果，如果不存在则使用空字符串
            coe_analysis = CertificateOfEmployment.get("openai_analysis", "")
            bs_analysis = BankStatements.get("openai_analysis", "")
            
            # 使用 KernelArguments 替代旧的 Context
            args = KernelArguments(
                COE=str(coe_analysis),
                BS=str(bs_analysis)
                )
            
            self.logger.info(f"准备调用 SK 函数 assess_credit，参数: {args}")
            # 调用函数
            result = await self.kernel.invoke(self.assess_credit, arguments=args)
            self.logger.info(f"SK 函数调用成功，结果: {result}")
            
            # 只返回 assessment 内容
            # return str(result)
            return {
                "status": "success",
                "assessment": str(result),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"SK 函数调用失败: {str(e)}", exc_info=True)
            return f"评估失败: {str(e)}"

async def credit_analysis_workflow(CertificateOfEmployment: str, BankStatements: str) -> str:
    # 初始化各客户端
    foundry_agent = FoundryIncomeAgent()
    blob_loader = IncomeProofLoader()
    evaluator = CreditEvaluator()

    try:
        # 验证输入参数
        if not CertificateOfEmployment or not BankStatements:
            raise ValueError("工作证明或银行流水文件名为空")
            
        # Step 1: 获取Blob路径
        COE_blob_path = await foundry_agent.get_income_blob_path(CertificateOfEmployment)
        BS_blob_path = await foundry_agent.get_income_blob_path(BankStatements)
        
        if not COE_blob_path or not BS_blob_path:
            raise ValueError("无法获取文件路径")
            
        print("COE_blob_path："+COE_blob_path)
        print("BS_blob_path："+BS_blob_path)
        
        # Step 2: 加载收入证明
        COE_data = await blob_loader.load_income_data(COE_blob_path)
        BS_data = await blob_loader.load_income_data(BS_blob_path)
        
        if not COE_data or not BS_data:
            raise ValueError("无法加载收入证明数据")
            
        print(COE_data)
        print(BS_data)
        
        # Step 3: 信用评估
        result = await evaluator.evaluate(COE_data, BS_data)
        print(result)
        return result
    except Exception as e:
        return f"工作流执行失败: {str(e)}"

# ------------------------- AutoGen 集成层（新增部分）-------------------------
class CreditAnalysisAgent(ConversableAgent):
    def __init__(self, name="CreditAnalyst"):
        llm_config = {
            "config_list": [{
                "model": "gpt-3.5-turbo",
                "api_key": "sk-anything", 
                "base_url": "http://placeholder.com"
            }],
            "timeout": 600
        }

        super().__init__(
            name=name,
            system_message="信用评分专家，可调用analyze_credit函数进行分析",
            human_input_mode="NEVER",
            llm_config=llm_config
        )

        # 定义函数并手动绑定到实例
        async def analyze_credit(CertificateOfEmployment: str, BankStatements: str) -> Dict[str, Any]:
            """执行信用评分分析（输入文件名，返回评分结果）"""
            return await self._analyze_credit_impl(CertificateOfEmployment,BankStatements)
        
        # 将函数绑定到实例
        self.analyze_credit = analyze_credit
        
        # 使用正确的注册方式
        self.register_for_llm(
            description="执行信用评分分析（输入文件名，返回评分结果）"
        )(analyze_credit)
        
        self.register_for_execution()(analyze_credit)
    
    async def _analyze_credit_impl(self, CertificateOfEmployment: str, BankStatements: str) -> Dict[str, Any]:
        """实际的信用分析实现"""
        result = await credit_analysis_workflow(CertificateOfEmployment,BankStatements)
        return {
            "origin_result": result,
            "auto_gen_compatible": True,
            "timestamp": datetime.now().isoformat()
        }

# 使用示例
async def demo():
    credit_agent = CreditAnalysisAgent()
    response = await credit_agent.analyze_credit("yinhangliushui.png","zhangsanzaizhi.png")  # 现在可以正确调用
    print("信用评分结果:", json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(demo())
