document.addEventListener('DOMContentLoaded', function() {
    const credentialsForm = document.getElementById('credentials-form');
    const fetchLogsBtn = document.getElementById('fetch-logs-btn');
    const predictBtn = document.getElementById('predict-btn');
    const launchForm = document.getElementById('launch-form');
    const launchSection = document.getElementById('launch-section');

    document.getElementById('verify-credentials-btn').addEventListener('click', async function() {
        try {
            const response = await fetch('/verify-credentials');
            const data = await response.json();
            
            if (data.success) {
                fetchLogsBtn.disabled = false;
                showMessage('logs-status', 'Creds verified Now fetch logs', true);
            } else {
                fetchLogsBtn.disabled = true;
                showMessage('logs-status', 'Invalid creds: ' + data.error, false);
            }
        } catch (error) {
            fetchLogsBtn.disabled = true;
            showMessage('logs-status', 'Error verifying creds: ' + error, false);
        }
    });

    credentialsForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const accessKey = document.getElementById('access-key').value;
        const secretKey = document.getElementById('secret-key').value;

        try {
            const response = await fetch('/set-credentials', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'access_key': accessKey,
                    'secret_key': secretKey
                })
            });

            const data = await response.json();
            if (data.message) {
                document.getElementById('verify-credentials-btn').disabled = false;
                showMessage('logs-status', 'Creds set success', true);
            }
        } catch (error) {
            showMessage('logs-status', 'Creds set failed:' + error, false);
        }
    });

    fetchLogsBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/fetch-logs');
            const data = await response.json();
            
            if (data.success) {
                predictBtn.disabled = false;
                showMessage('logs-status', 'Cloudwatch logs fetch success and saved', true);
            } else {
                predictBtn.disabled = true;
                showMessage('logs-status', data.error, false);
            }
        } catch (error) {
            predictBtn.disabled = true;
            showMessage('logs-status', 'Error fetching logs: ' + error, false);
        }
    });

    predictBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/predict');
            const data = await response.json();
            
            if (data.success) {
                const message = `Predicted CPU Util % in the next 15 min: ${data.predicted_max_cpu.toFixed(2)}%<br>`;
                showMessage('prediction-result', message, true);
                launchSection.style.display = 'block';
                showMessage('prediction-result', message + 'Recommended to Launch new instance. You can launch instance from below form.', true);
            } else {
                showMessage('prediction-result', 'Prediction failed: ' + data.error, false);
            }
        } catch (error) {
            showMessage('prediction-result', 'Error making prediction: ' + error, false);
        }
    });

    launchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const amiId = document.getElementById('ami-id').value;
        const instanceType = document.getElementById('instance-type').value;

        try {
            const response = await fetch('/launch-instance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'ami_id': amiId,
                    'instance_type': instanceType
                })
            });

            const data = await response.json();
            if (data.success) {
                showMessage('launch-status', `Instance launched successInstance ID: ${data.instance_id}`, true);
            } else {
                showMessage('launch-status', 'Instance launch failed:' + data.error, false);
            }
        } catch (error) {
            showMessage('launch-status', 'launching instance failed: ' + error, false);
        }
    });

    function showMessage(elementId, message, isSuccess) {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast ${isSuccess ? 'success' : 'error'}`;
        toast.innerHTML = message;
        
        // Add to container
        const container = document.getElementById('toast-container');
        container.appendChild(toast);

        // Remove toast after 5 seconds
        setTimeout(() => {
            toast.classList.add('toast-fade-out');
            setTimeout(() => {
                container.removeChild(toast);
            }, 500);
        }, 5000);

        // Also update the original status element if it exists
        const statusElement = document.getElementById(elementId);
        if (statusElement) {
            statusElement.innerHTML = message;
            statusElement.className = isSuccess ? 'success-message' : 'error-message';
        }
    }
});
