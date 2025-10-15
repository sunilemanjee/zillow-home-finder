# Zillow Home Finder - Chainlit App

A modern AI-powered real estate assistant built with Chainlit and integrated with your custom LLM endpoint.

## Features

- 🏠 Real estate property search and analysis
- 💬 Interactive chat interface powered by Chainlit
- 🤖 AI assistant with custom LLM integration
- 📊 Property insights and market analysis
- 🎨 Modern, responsive UI

## Setup

### 1. Install Dependencies

```bash
cd app
pip install -r requirements.txt
```

### 2. Environment Configuration

The app automatically loads configuration from the parent directory's `variables.env` file. Make sure the following variables are set:

- `LLM_URL`: Your custom LLM endpoint URL
- `LLM_API_KEY`: Your API key
- `LLM_MODEL`: The model name to use

### 3. Run the Application

```bash
# Option 1: Using Chainlit CLI
chainlit run app.py --port 8022

# Option 2: Direct Python execution
python app.py
```

The app will be available at `http://localhost:8022`

## Usage

1. Open your browser and navigate to `http://localhost:8022`
2. Start chatting with the AI assistant about real estate
3. Ask questions about:
   - Property searches
   - Market analysis
   - Neighborhood insights
   - Real estate processes

## Customization

### UI Customization

Edit `.chainlit/config.toml` to customize:
- App name and description
- Theme colors
- Feature toggles
- UI behavior

### Adding Features

The main app logic is in `app.py`. You can extend it by:
- Adding new message handlers
- Creating custom actions
- Integrating with additional APIs
- Adding file upload capabilities

## Architecture

- **Frontend**: Chainlit UI framework
- **Backend**: Python with async/await support
- **LLM Integration**: OpenAI-compatible API client
- **Configuration**: Environment variables via python-dotenv

## Troubleshooting

### Common Issues

1. **Connection Error**: Verify your LLM_URL and API_KEY in variables.env
2. **Import Errors**: Ensure all dependencies are installed via requirements.txt
3. **Port Conflicts**: Change the port in the run command if 8022 is occupied

### Debug Mode

Run with debug logging:
```bash
chainlit run app.py --debug
```

## Development

To extend the application:

1. Add new handlers in `app.py`
2. Update `requirements.txt` for new dependencies
3. Modify `.chainlit/config.toml` for UI changes
4. Test with `chainlit run app.py --debug`
