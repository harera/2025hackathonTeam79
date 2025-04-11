from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import json
import requests
import asyncio
from typing import Dict, Any
from datetime import datetime

# Azure 服务SDK
from azure.storage.blob.aio import BlobServiceClient
from azure.cosmos.aio import CosmosClient

# Semantic Kernel
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig

# --- Cosmos DB 客户端 ---
class CosmosDBClient:
    """封装与Cosmos DB的交互（使用Connection String连接）"""
    
    def __init__(self):
        self.client = CosmosClient.from_connection_string(
            "AccountEndpoint=https://hackaloandb.documents.azure.com:443/;AccountKey=nBAAGrW3NayW8oLRgPh4LgJNE6aqidQRFsU12WVsOwjuOUYHFQGh3HyPPeTRYzVUwON1Gj3KE9SIACDbAsD3zQ==;"
        )
    
    async def get_user_info(self, applicantId: str) -> Dict[str, Any]:
        """根据user_id获取用户信息"""
        try:
            database = self.client.get_database_client("cosmicworks")
            container = database.get_container_client("userinfo")
            
            # 使用参数化查询（更安全）
            query = "SELECT * FROM c WHERE c.id = @applicantId"
            parameters = [
                {"name": "@applicantId", "value": applicantId}
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
                raise ValueError(f"未找到用户ID: {applicantId}")
            
            return items[0]
        except Exception as e:
            print(f"[Cosmos DB] 查询失败: {str(e)}")
            raise
        finally:
            await self.client.close()


# --- Azure Blob 存储操作 ---
class ContractProofLoader:
    """从Blob Storage加载收入证明"""
    
    def __init__(self):
        self.client = BlobServiceClient(
            account_url="https://hackathonfound1559274341.blob.core.windows.net",
            credential="Foer+fIx7QM9ihnWxnwJrnRx/GA3YydT3UJ4vzVwr4xkfAdHoUacaOqx5CgVwt07vrgLl1N2IA3l+AStAfISnA=="
        )
    async def load_contract_data(self, blob_path: str) -> Dict[str, Any]:
        """异步加载JSON格式的收入证明"""
        try:
            blob_client = self.client.get_blob_client(
                container="contract",
                blob=blob_path
            )
            # 关键修正点：添加await
            downloader = await blob_client.download_blob()
            data = await downloader.readall()
            #data = await blob_client.download_blob().readall()
            return data #json.loads(data.decode('utf-8'))
            
        except Exception as e:
            print(f"[Blob Storage] 加载失败: {blob_path}, 错误: {str(e)}")
            raise

class ComplianceReview:
    
    def __init__(self):
        self.kernel = sk.Kernel()
        self.kernel.add_service(
            AzureChatCompletion(
                service_id="credit_ai",
                deployment_name="gpt-4o-mini",
                endpoint="https://hackathonfound6454649233.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                api_key="U4fD1XnytGhxvUlsRYSRL472aImrcK17LbNS5CQfGOTRpkWKAwpzJQQJ99BDACHYHv6XJ3w3AAAAACOGIngS",
                api_version="2024-12-01-preview"
            )
        )
        
        self.compliance_review = self.kernel.add_function(
            plugin_name="complianceReview",
            function_name="compliance_review",
            prompt_template_config=PromptTemplateConfig(
            template="""
                You are a professional pre-approval specialist for bank mortgage contracts, responsible for reviewing the compliance of contracts in accordance with the bank's internal rules and regulations before the contract is issued.
                You have user data {{$user_Info}},
                Key information about the contract {{$contract}}
                The following information is mainly reviewed：
                Basic Borrower Qualifications
                1.Age Requirements:
                Applicants must be at least 18 years old, and their age at the time of loan maturity must not exceed 60 years old (inclusive).
                For premium clients or special projects, exceptions may be granted up to 65 years old upon approval from the head office, provided that a co-borrower or additional collateral is secured.
                2.Credit History:
                No more than 3 consecutive or 6 cumulative late payments within the past 2 years.
                No major negative credit records (e.g., bad debts, debt settlements).
                3.Repayment Capacity:
                Monthly income must be at least twice the monthly installment (including other liabilities).
                4.The loan amount shall not exceed 70% of the property's appraised value (for first homes) or 50% (for second homes)
                You need to clearly answer whether the contract is compliant or not, and then generate a proposal for the loan contract in no more than 3 sentences.
                Focus on contract compliance!                
                """
            )
        )

    async def evaluate(self, UserInfo: Dict[str, Any],Contract: Dict[str, Any],LoanPolicy: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 使用 KernelArguments 替代旧的 Context
            args = KernelArguments(
                user_Info=str(UserInfo),
                contract=str(Contract),
                loan_Policy = str(LoanPolicy)
                )
            
            # 调用函数
            result = await self.kernel.invoke(self.compliance_review, arguments=args)
            
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

async def compliance_review_workflow(user_id: str) -> Dict[str, Any]:
    # 初始化各客户端
    cosmos_client = CosmosDBClient()
    blob_loader = ContractProofLoader()
    evaluator = ComplianceReview()

    try:
        # Step 1: 获取用户信息
        user_info = await cosmos_client.get_user_info(user_id)
        
        borrower_name = user_info.get("name")
        borrower_id = str(user_id)
        contract_number = "CN-20250408-004"
        loan_amount = str(user_info.get("loanAmount"))
        loan_term_years = str(user_info.get("loanTerm"))
        lst = user_info.get("loanTerm")
        loan_start_date_first = datetime.fromisoformat(user_info.get("loanStartDate")).date().strftime("%Y-%m-%d")#(user_info.get("loanStartDate")).date()
        loan_start_date = str(loan_start_date_first)
        loan_end_date_first = datetime.fromisoformat(user_info.get("loanStartDate")).date() + relativedelta(years=25)
        loan_end_date = str(loan_end_date_first.strftime("%Y-%m-%d"))
        loan_interest_rate = "2.35%"
        repayment_method = "Fixed Payment Mortgage"
        repayment_due_date = "3rd day of each month"
        property_address = "Room 1803, Building 5, NAGA Shangyuan (或 NAGA Upper Court), No. 9 Dongzhimennei Street, Dongcheng District, Beijing, China"
        property_area = str(user_info.get("propertyArea")) 
        property_price = int(user_info.get("propertyPrice"))
        ltv_ratio = str("{:.2%}".format(user_info.get("loanAmount")/property_price)) #总房款暂时设置成两百万，等有了前端传入的数据再修改

        contractData = {
            "borrower_name": borrower_name,
            "borrower_id": borrower_id,
            "contract_number": contract_number,
            "loan_amount": loan_amount,
            "loan_term_years": loan_term_years,
            "loan_start_date": loan_start_date,
            "loan_end_date": loan_end_date,
            "loan_interest_rate": loan_interest_rate,
            "repayment_method": repayment_method,
            "repayment_due_date": repayment_due_date,
            "property_address": property_address,
            "property_area": property_area,
            "ltv_ratio": ltv_ratio
        }

        contractData_json = json.dumps(contractData, ensure_ascii=False, indent=2)
        
        web_app_url = "https://pdfcontract-d6hef0cgg0dughhq.eastus2-01.azurewebsites.net/generate_mortgage_contract"
        # 调用 POST 方法
        response = requests.post(web_app_url, json=contractData)
        contract_file_path = response.json()["filename"]
        print("Contract documents generated:" + contract_file_path)
        contract = await blob_loader.load_contract_data(contract_file_path)
        loanPolicy = await blob_loader.load_contract_data("Bank Internal Personal Housing Loan Policy.pdf")
        print("It is contained in the bank's internal contract statute : Bank Internal Personal Housing Loan Policy.pdf")
        # Step 3: 合规性审查
        print("The agent has received the contract and specification documents and is reviewing for compliance")
        result = await evaluator.evaluate(user_info,contractData,loanPolicy)

        print(result)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"工作流执行失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
    
if __name__ == "__main__":
    async def main():
        userId = "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca"
        try:
            result = await compliance_review_workflow(userId)
            print("\n最终结果:")
            print(result)
        except Exception as e:
            print(f"主程序错误: {str(e)}")

    asyncio.run(main())