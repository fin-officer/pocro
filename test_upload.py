import requests
import sys

def test_upload():
    url = 'http://localhost:9000/process-invoice'
    file_path = '/home/tom/github/wronai/pocro/invoice.pdf'
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': ('invoice_en.pdf', f, 'application/pdf')}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(response.json())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_upload()
