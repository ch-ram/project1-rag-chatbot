"""
Frontend styling.
"""


def load_css():

    return """
    <style>

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .main > div {
        max-width: 1400px;
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .app-header {
        margin-bottom: 2rem;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 1rem;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 0.5rem;
    }

    section[data-testid="stSidebar"] {
        background: #FAFAFA;
        border-right: 1px solid #E5E7EB;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #D1D5DB;
        border-radius: 14px;
        padding: 1rem;
        background: white;
    }

    .chat-user {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }

    .chat-assistant {
        background: white;
        padding: 1rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    .message-role {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #111827;
    }

    .message-content {
        line-height: 1.7;
        color: #111827;
    }

    .source-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        background: white;
    }

    .source-title {
        font-weight: 600;
        color: #111827;
    }

    .source-meta {
        color: #6B7280;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    .status-connected {
        background: #ECFDF5;
        color: #065F46;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        display: inline-block;
        border: 1px solid #D1FAE5;
    }

    .status-offline {
        background: #FEF2F2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        display: inline-block;
        border: 1px solid #FECACA;
    }

    .empty-state {
        border: 1px dashed #D1D5DB;
        border-radius: 18px;
        padding: 4rem 2rem;
        text-align: center;
        background: #FAFAFA;
        margin-top: 2rem;
    }

    .empty-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
    }

    .empty-text {
        color: #6B7280;
        line-height: 1.8;
    }

    </style>
    """