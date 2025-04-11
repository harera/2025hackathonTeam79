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
import azure.functions as func

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

from autogen import ConversableAgent

# 加载环境变量（适用于本地开发和Azure Function配置）
load_dotenv()


# --- Foundry 客户端 ---
class FoundryIncomeAgent:
    """封装与Foundry AI的交互"""
    
    def __init__(self):
        self.client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str="",
        )  
        # print(self.client)
    async def get_income_blob_path(self, filename: str) -> str:
        """
        调用Foundry Agent获取收入证明的Blob路径
        """    
        try:
            agent_definition = await self.client.agents.get_agent(agent_id="")
            getBlobPath_agent = AzureAIAgent(client=self.client, definition=agent_definition)
            thread: AzureAIAgentThread = None
            response = await getBlobPath_agent.get_response(messages=filename, thread=thread)
            res = response.message.items[0].text
            # print(f"# {response.name}: {response}")
            # print(response.message.items[0].text)
            # print(f"Full response: {vars(response)}")  # 查看所有属性
            # print(f"Content type: {type(response)}")  # 确认类型
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
        self.client = BlobServiceClient(
            account_url="",
            credential=""
        )
    
    async def load_income_data(self, blob_path: str) -> Dict[str, Any]:
        """异步加载JSON格式的收入证明"""
        try:
            blob_client = self.client.get_blob_client(
                container="ocrjson",
                blob=blob_path
            )
            # 关键修正点：添加await 
            downloader = await blob_client.download_blob()
            data = await downloader.readall()
            #data = await blob_client.download_blob().readall()
            return json.loads(data.decode('utf-8'))
            
        except Exception as e:
            print(f"[Blob Storage] 加载失败: {blob_path}, 错误: {str(e)}")
            raise
    
# --- 信用评估核心 ---
class CreditEvaluator:
    """极简信用评估（仅基于月收入）"""
    
    def __init__(self):
        self.kernel = sk.Kernel()
        self.kernel.add_service(
            AzureChatCompletion(
                service_id="credit_ai",
                deployment_name="gpt-4o-mini",
                endpoint="",
                api_key="",
                api_version="2024-12-01-preview"
            )
        )
        
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
    
    async def evaluate(self, CertificateOfEmployment: Dict[str, Any],BankStatements: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 使用 KernelArguments 替代旧的 Context
            args = KernelArguments(
                COE=str(CertificateOfEmployment.get("openai_analysis", 0)),
                BS=str(BankStatements.get("openai_analysis", 0))
                )
            
            # 调用函数
            result = await self.kernel.invoke(self.assess_credit, arguments=args)
            
            return {
                "status": "success",
                "assessment": str(result),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="analyze_credit", methods=["POST"])
async def analyze_credit(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP触发的信用评分分析函数"""
    try:
        # 解析请求体
        req_body = req.get_json()
        certificate_file = req_body.get("certificate_file")
        statements_file = req_body.get("statements_file")
        
        if not certificate_file or not statements_file:
            return func.HttpResponse(
                "请提供certificate_file和statements_file参数",
                status_code=400
            )
        
        # 调用现有工作流
        result = await credit_analysis_workflow(certificate_file, statements_file)
        
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )
        
    except json.JSONDecodeError:
        return func.HttpResponse(
            "无效的JSON请求体",
            status_code=400
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

async def credit_analysis_workflow(CertificateOfEmployment: str, BankStatements: str) -> Dict[str, Any]:
    # 初始化各客户端
    foundry_agent = FoundryIncomeAgent()
    blob_loader = IncomeProofLoader()
    evaluator = CreditEvaluator()

    try:
        # Step 1: 获取Blob路径
        COE_blob_path = await foundry_agent.get_income_blob_path(CertificateOfEmployment)
        print("COE_blob_path："+COE_blob_path)
        BS_blob_path = await foundry_agent.get_income_blob_path(BankStatements)
        print("BS_blob_path："+BS_blob_path)
        # Step 2: 加载收入证明
        COE_data = await blob_loader.load_income_data(COE_blob_path)
        print("The content of the certificate of employment:" + COE_data)
        BS_data = await blob_loader.load_income_data(BS_blob_path)
        print("The contents of the bank statement:" + BS_data)
        # Step 3: 信用评估
        result = await evaluator.evaluate(COE_data,BS_data)
        print(result)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"工作流执行失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

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
    response = await credit_agent.analyze_credit("zhangsanzaizhi.png","yinhangliushui.png")  # 现在可以正确调用
    print("Credit score results:", json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(demo())