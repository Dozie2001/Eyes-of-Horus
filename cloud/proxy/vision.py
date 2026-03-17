"""
Vision API proxy for Eyes of Horus.

Edge devices send images here. We forward to Groq/Gemini using OUR API keys,
so the customer never sees them.

Routes:
    POST /proxy/vision/describe — single image description
    POST /proxy/vision/describe-video — multi-frame video description
"""

# TODO: Implement vision proxy endpoints
# - Validate edge device API key
# - Forward image to Groq/Gemini
# - Return description
