import asyncio
from loan_application_system import LoanApplication

class LoanApplicationAdapter:
    """适配器类，使现有系统更好地配合API使用"""
    
    @staticmethod
    async def evaluate_loan_async(loan_data):
        """异步评估贷款，用于API调用"""
        # 运行在单独的线程中以避免阻塞API服务器
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, LoanApplicationAdapter._evaluate_loan_sync, loan_data)
        return result
    
    @staticmethod
    def _evaluate_loan_sync(app):
        """同步评估贷款（在单独线程中运行）"""
        # # 初始化应用
        # app = LoanApplication()
        
        # # 设置数据
        # app.collected_data = loan_data
        
        # 执行评估
        app.phase_evaluation()
        
        # 返回结果
        return {
            "evaluation_result": app.evaluation_results,
            "expert_results": app.expert_results if hasattr(app, "expert_results") else {},
            "status": "success" if "未能获取" not in app.evaluation_results and "评估失败" not in app.evaluation_results else "failed"
        } 