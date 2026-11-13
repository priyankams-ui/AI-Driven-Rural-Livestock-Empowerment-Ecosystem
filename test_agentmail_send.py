import os
import sys
from dotenv import load_dotenv
from agentmail import AgentMail
from agentmail.core.api_error import ApiError

def test_send():
    load_dotenv()
    
    api_key = (os.getenv("AGENTMAIL_API_KEY") or "").strip()
    inbox_id = (os.getenv("AGENTMAIL_INBOX_ID") or "").strip()
    
    # 1. Diagnostics (Safe, no secrets)
    api_key_configured = bool(api_key and 'your_agentmail' not in api_key.lower() and 'placeholder' not in api_key.lower())
    inbox_configured = bool(inbox_id and 'your_agentmail' not in inbox_id.lower() and 'placeholder' not in inbox_id.lower())
    
    print("=== AgentMail Configuration Diagnostics ===")
    print(f"API key configured: {api_key_configured}")
    print(f"Inbox configured: {inbox_configured}")
    print(f"Inbox ID: {inbox_id}")
    
    if not api_key_configured or not inbox_configured:
        print("API key valid: False")
        print("Send permission: False")
        print("Configuration incomplete.")
        return False
        
    client = AgentMail(api_key=api_key)
    
    # Verify API key validity and inbox existence
    api_key_valid = False
    inbox_verified = False
    try:
        inbox_obj = client.inboxes.get(inbox_id=inbox_id)
        api_key_valid = True
        inbox_verified = True
        print(f"API key valid: {api_key_valid}")
        print(f"Inbox verified: {inbox_verified} ({inbox_obj.email})")
    except ApiError as e:
        status_code = getattr(e, 'status_code', None)
        print(f"API key valid: False (API error HTTP {status_code}: {e})")
        print("Send permission: False")
        return False
    except Exception as e:
        print(f"API key valid: False (Error: {e})")
        print("Send permission: False")
        return False
        
    # Recipient for single test email
    recipient = "priyankams738@gmail.com"
    if len(sys.argv) > 1 and '@' in sys.argv[1]:
        recipient = sys.argv[1].strip()
    elif os.environ.get("TEST_RECIPIENT") and '@' in os.environ.get("TEST_RECIPIENT"):
        recipient = os.environ.get("TEST_RECIPIENT").strip()
        
    subject = "PashuDhan AI Email Test"
    body = "This is a test email from the PashuDhan AI application."
    
    print("\n=== Single Test Email Dispatch ===")
    print(f"From Inbox: {inbox_id}")
    domain = recipient.split('@')[-1]
    print(f"To Domain: @{domain}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    
    try:
        response = client.inboxes.messages.send(
            inbox_id=inbox_id,
            to=recipient,
            subject=subject,
            text=body
        )
        print("Send permission: True")
        print(f"AgentMail send response: SUCCESS (Response: {response})")
        return True
    except ApiError as e:
        status_code = getattr(e, 'status_code', None)
        err_body = getattr(e, 'body', None)
        err_msg = ""
        if hasattr(err_body, 'message') and err_body.message:
            err_msg = str(err_body.message)
        elif isinstance(err_body, dict) and 'message' in err_body:
            err_msg = str(err_body['message'])
        else:
            err_msg = str(err_body or e)
            
        print(f"AgentMail API Error (HTTP {status_code}): {err_msg}")
        
        # Check permissions vs spam
        if status_code in (401,):
            print("Send permission: False (Unauthorized)")
        elif 'permission' in err_msg.lower() or 'forbidden' in err_msg.lower():
            print("Send permission: False")
        else:
            # Reached inbox and evaluated by policy
            print("Send permission: True (API key authorized for inbox, but policy/content check triggered)")
            
        if "spam" in err_msg.lower() or "rejected" in err_msg.lower():
            print("\n[RESULT] Message rejected by AgentMail's spam protection.")
            print(f"Exact error: {err_msg}")
        return False
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    test_send()
