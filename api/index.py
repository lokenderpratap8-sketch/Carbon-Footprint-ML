from app import app

# Vercel serverless function handler
def handler(event, context):
    # For Vercel serverless functions
    from mangum import Mangum
    asgi_app = Mangum(app)
    return asgi_app(event, context)
