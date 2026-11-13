import os
import sys
from dotenv import load_dotenv
from agentmail import AgentMail
from agentmail.core.api_error import ApiError

def main():
    # 1. Load .env
    load_dotenv()

    api_key = (os.getenv("AGENTMAIL_API_KEY") or "").strip()
    inbox_id = (os.getenv("AGENTMAIL_INBOX_ID") or "").strip()

    # 2 & 3. Verify AGENTMAIL_API_KEY and AGENTMAIL_INBOX_ID exist without printing them
    has_api_key = bool(api_key and 'your_agentmail' not in api_key.lower() and 'placeholder' not in api_key.lower())
    has_inbox_id = bool(inbox_id and 'your_agentmail' not in inbox_id.lower() and 'placeholder' not in inbox_id.lower())

    config_detected = has_api_key and has_inbox_id

    # Print AgentMail configuration detected: YES/NO
    print(f"AgentMail configuration detected: {'YES' if config_detected else 'NO'}")

    if not config_detected:
        print("AgentMail authentication: FAIL")
        print("Email send: FAIL")
        if not has_api_key:
            print("Notice: AGENTMAIL_API_KEY is not configured or is using placeholder in .env")
        if not has_inbox_id:
            print("Notice: AGENTMAIL_INBOX_ID is not configured or is using placeholder in .env")
        print("Please configure real AGENTMAIL_API_KEY and AGENTMAIL_INBOX_ID in .env from https://agentmail.to")
        return False

    # 5. Determine recipient email from argument, environment, terminal input, or default
    recipient = None
    if len(sys.argv) > 1 and '@' in sys.argv[1]:
        recipient = sys.argv[1].strip()
    elif os.environ.get("TEST_RECIPIENT") and '@' in os.environ.get("TEST_RECIPIENT"):
        recipient = os.environ.get("TEST_RECIPIENT").strip()
    elif sys.stdin and sys.stdin.isatty():
        try:
            user_input = input("Enter recipient email (press Enter for priyankams738@gmail.com): ").strip()
            if user_input and '@' in user_input:
                recipient = user_input
        except Exception:
            pass

    if not recipient:
        # Documented test recipient
        recipient = "priyankams738@gmail.com"

    # 4. Send a test email through AgentMail
    subject = "AgentMail Verification Test - Livestock Marketplace"
    text_body = "Hello! This is a test email sent from Livestock Marketplace via AgentMail Free API."
    html_body = """
    <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px;">
        <h2 style="color: #166534;">Livestock Marketplace</h2>
        <p>This is a test email verifying that <strong>AgentMail</strong> is successfully configured for real email delivery.</p>
        <p>All OTP verification, purchase confirmation, and livestock availability alerts will be delivered via AgentMail.</p>
    </div>
    """

    auth_passed = False
    send_passed = False
    message_id = None

    try:
        client = AgentMail(api_key=api_key)

        # Attempt to verify inbox or send message
        response = client.inboxes.messages.send(
            inbox_id=inbox_id,
            to=recipient,
            subject=subject,
            text=text_body,
            html=html_body
        )

        auth_passed = True
        send_passed = True
        if hasattr(response, 'message_id'):
            message_id = response.message_id
        elif isinstance(response, dict):
            message_id = response.get('message_id')

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

        if status_code in (401, 403):
            auth_passed = False
            send_passed = False
            print(f"Error detail: Authentication rejected (HTTP {status_code})")
        else:
            auth_passed = True  # Key was authenticated, but operation had an issue (e.g. 404 inbox or 422)
            send_passed = False
            print(f"Error detail: API returned HTTP {status_code}: {err_msg}")

    except Exception as e:
        auth_passed = False
        send_passed = False
        print(f"Error detail: Network/Connection exception ({type(e).__name__}): {e}")

    # 6. Print required lines
    print(f"AgentMail authentication: {'PASS' if auth_passed else 'FAIL'}")
    print(f"Email send: {'PASS' if send_passed else 'FAIL'}")
    if message_id:
        print(f"Provider message ID: {message_id}")

    return send_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
