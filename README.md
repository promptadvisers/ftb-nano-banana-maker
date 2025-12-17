# FTB Nano Banana Maker 🍌

A web-based image generation tool powered by Google's Gemini image models. Features intelligent prompt creation and enhancement using best practices from the Nano Banana Pro prompting guide.

![Demo](https://img.shields.io/badge/Gemini-Image_Generation-yellow)

## Features

- **Quick Idea to Prompt** - Enter a basic idea like "cat in space" and get a fully crafted prompt using AI
- **Prompt Enhancement** - Improve existing prompts with professional photography and cinematography techniques
- **Dual Model Support** - Choose between Flash (fast) and Pro (4K quality) image generation
- **Persistent API Key** - Your Gemini API key is stored locally in the browser

## Models

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash-image` | Fast, cost-effective (~$0.04/image) |
| `gemini-3-pro-image-preview` | Advanced features, up to 4K resolution |

Prompt creation and enhancement uses `gemini-2.5-flash`.

## Setup

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com)
2. Clone and serve:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ftb-nano-banana-maker.git
   cd ftb-nano-banana-maker
   python3 -m http.server 8080
   ```
3. Open http://localhost:8080
4. Enter your API key and start generating

## Prompting Best Practices

This tool uses the Nano Banana Pro prompting methodology:

- **Natural Language** - Full sentences, not comma-separated tags
- **Context is King** - Include intended use case for better artistic inference
- **Specificity & Materiality** - Define textures to avoid the "plastic AI look"
- **Ultimate Structure** - Subject → Action → Environment → Lighting → Technical → Context

## Files

- `index.html` - Complete app (HTML/CSS/JS)
- `gemini_api_documentation_december_2025.md` - Gemini API reference
- `nano-banana-pro-prompting-guide.md` - Prompting best practices

## License

MIT
