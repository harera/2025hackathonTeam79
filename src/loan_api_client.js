/**
 * Solvay Capital Bank Loan Evaluation System - Frontend API Call Example
 */

// API Base URL - Use relative path
const API_BASE_URL = '';

/**
 * Submit loan application and get evaluation results
 * @param {Object} formData - Loan application form data
 * @returns {Promise<Object>} - Evaluation results
 */
async function submitLoanApplication(formData) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/loan/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: formData.name,
        age: formData.age,
        phone: formData.phone,
        email: formData.email,
        address: formData.address,
        employer: formData.employer,
        position: formData.position,
        monthly_income: formData.monthlyIncome,
        loan_amount: formData.loanAmount,
        loan_purpose: formData.loanPurpose,
        loan_term: formData.loanTerm,
        property_address: formData.propertyAddress,
        loan_start_date: formData.loanStartDate,
        property_size: formData.propertySize
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unable to parse error response' }));
      throw new Error(`API Error: ${response.status} - ${errorData.detail || 'Unknown error'}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Loan application submission failed:', error);
    throw error;
  }
}

/**
 * Send chat message and get response
 * @param {string} message - User message
 * @param {string} sessionId - Session ID (optional)
 * @param {string} inputMode - Input mode ('chat' or 'form')
 * @param {Object} formData - Form data (used in form mode)
 * @returns {Promise<Object>} - Chat response
 */
async function sendChatMessage(message, sessionId = null, inputMode = 'chat', formData = null) {
  try {
    const requestData = {
      session_id: sessionId,
      message: message,
      input_mode: inputMode,
    };
    
    // Add form data (if exists)
    if (inputMode === 'form' && formData) {
      requestData.form_data = formData;
    }
    
    console.log('Sending chat request:', requestData);
    
    const response = await fetch(`${API_BASE_URL}/api/loan/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    console.log('API response status:', response.status);
    
    if (!response.ok) {
      // Attempt to get error details, but not guaranteed to succeed
      const errorData = await response.json().catch(() => ({ detail: 'Unable to parse error response' }));
      throw new Error(`API Error: ${response.status} - ${errorData.detail || 'Unknown error'}`);
    }

    const data = await response.json();
    console.log('API response data:', data);
    return data;
  } catch (error) {
    console.error('Chat message sending failed:', error);
    throw error;
  }
}

/**
 * Get chat history
 * @param {string} sessionId - Session ID
 * @returns {Promise<Array>} - Chat history
 */
async function getChatHistory(sessionId) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/loan/chat-history/${sessionId}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unable to parse error response' }));
      throw new Error(`API Error: ${response.status} - ${errorData.detail || 'Unknown error'}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Failed to get chat history:', error);
    throw error;
  }
}

/**
 * Get session state
 * @param {string} sessionId - Session ID
 * @returns {Promise<Object>} - Session state
 */
async function getSessionState(sessionId) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/loan/session/${sessionId}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unable to parse error response' }));
      throw new Error(`API Error: ${response.status} - ${errorData.detail || 'Unknown error'}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Failed to get session state:', error);
    throw error;
  }
}

/**
 * Upload loan-related documents
 * @param {string} userId - User ID
 * @param {File} employmentCert - Employment certificate file
 * @param {File} bankStatement - Bank statement file
 * @returns {Promise<Object>} - Upload result
 */
async function uploadDocuments(userId, employmentCert, bankStatement) {
  const formData = new FormData();
  formData.append('user_id', userId);
  
  if (employmentCert) {
    formData.append('employment_certificate', employmentCert);
  }
  
  if (bankStatement) {
    formData.append('bank_statement', bankStatement);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/loan/upload-documents`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unable to parse error response' }));
      throw new Error(`API Error: ${response.status} - ${errorData.detail || 'Unknown error'}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Document upload failed:', error);
    throw error;
  }
}

/**
 * Check API service health status
 * @returns {Promise<boolean>} - Is the service healthy
 */
async function checkApiHealth() {
  try {
    console.log('Checking API health status...');
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
    });
    console.log('Health check response status:', response.status);
    
    if (response.ok) {
      const data = await response.json();
      console.log('Health check response:', data);
      return data.status === 'healthy';
    }
    return false;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
}

// Usage example
document.addEventListener('DOMContentLoaded', async () => {
  // Example: Check API health status
  const isHealthy = await checkApiHealth();
  console.log('API service status:', isHealthy ? 'Normal' : 'Abnormal');
  
  // The following is a complete process example code, actual usage can be adjusted as needed
  /* 
  // Step 1: Upload documents
  const uploadResult = await uploadDocuments(
    '08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca',
    document.getElementById('employment-cert-input').files[0],
    document.getElementById('bank-statement-input').files[0]
  );
  console.log('Document upload result:', uploadResult);
  
  // Step 2: Submit loan application and get evaluation
  const loanResult = await submitLoanApplication({
    name: 'Zhang San',
    age: '30',
    phone: '1234567890',
    email: 'zhangsan@example.com',
    address: 'Haidian District, Beijing',
    employer: 'Solvay Capital Bank',
    position: 'Software Engineer',
    monthlyIncome: '30000',
    loanAmount: '2000000',
    loanPurpose: 'Purchase Property',
    loanTerm: '30',
    propertyAddress: 'Haidian District, Beijing',
    loanStartDate: '2023-12-01',
    propertySize: '120'
  });
  console.log('Loan evaluation result:', loanResult);
  
  // Handle evaluation result
  if (loanResult.status === 'success') {
    // Display evaluation result
    document.getElementById('result-container').innerHTML = 
      `<h3>Evaluation Result</h3><pre>${loanResult.evaluation_result}</pre>`;
  } else {
    // Display error
    document.getElementById('result-container').innerHTML = 
      `<h3>Evaluation Failed</h3><p>${loanResult.evaluation_result || 'Failed to get evaluation result'}</p>`;
  }
  */
}); 