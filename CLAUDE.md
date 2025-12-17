# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a documentation reference repository containing Google Gemini API documentation (December 2025). The main file `gemini_api_documentation_december_2025.md` covers three primary capabilities:

1. **Video Understanding** - Processing and analyzing video content with Gemini models
2. **Image Generation** - Creating images using Gemini native generation and Imagen models
3. **Real-time Voice (Live API)** - Bidirectional audio streaming for voice conversations

## Key Models Referenced

- **Gemini 3 Pro** (`gemini-3-pro-preview`) - Flagship model for complex reasoning
- **Gemini 3 Pro Image** (`gemini-3-pro-image-preview`) - Advanced image generation up to 4K (codename "Nano Banana Pro")
- **Gemini 2.5 Flash** (`gemini-2.5-flash`) - Production-recommended for video understanding
- **Gemini 2.5 Flash Image** (`gemini-2.5-flash-image`) - Fast, cost-effective image generation
- **Native Audio** (`gemini-2.5-flash-native-audio-preview-09-2025`) - Real-time voice conversations

## SDK

All examples use the unified `google-genai` Python SDK:
```bash
pip install -U google-genai pillow pyaudio
```

## Quick Reference Sections

- Installation & Authentication: Lines 18-50
- Video Understanding: Lines 83-286
- Image Generation: Lines 289-568
- Real-time Voice/Live API: Lines 571-919
- Rate Limits & Pricing: Lines 922-964
