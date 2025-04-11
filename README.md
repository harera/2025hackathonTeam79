# AutoGen Loan Application System

A simple loan application system built with AutoGen that demonstrates multi-agent conversations for loan processing.

## System Overview

The system has two phases:
1. **Data Collection Phase**: A conversational agent collects loan application data from the user.
2. **Evaluation Phase**: A hierarchical agent system with a main decision agent and three specialized agents (Credit, Fraud, Compliance) evaluates the application.

## Requirements

- Python 3.11.9
- Dependencies listed in requirements.txt

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure your `.env` file contains the following environment variables for Azure OpenAI:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_DEPLOYMENT=your_deployment
   AZURE_OPENAI_API_VERSION=your_api_version
   ```

## Running the Application

```
python loan_application_system.py
```

## Usage

1. During the data collection phase, answer the questions asked by the data collection agent.
2. When all required information is collected, the agent will display "DATA_COLLECTION_COMPLETE".
3. The system will then proceed to the evaluation phase automatically, where multiple specialized agents will analyze the application.
4. The final decision will be displayed at the end of the process.

## Agent Details

- **Data Collection Agent**: Collects all necessary information for the loan application
- **Decision Agent**: Main agent that coordinates with specialized agents and makes the final decision
- **Credit Evaluation Agent**: Analyzes credit-related information
- **Fraud Detection Agent**: Detects potential fraud indicators
- **Compliance Review Agent**: Ensures compliance with regulations

## Notes

- This is a demonstration system and should not be used for real loan applications.
- The agents' responses are based on the provided information and do not represent actual financial advice. 