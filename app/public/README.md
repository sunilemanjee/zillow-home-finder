# Custom Assets for Zillow Home Finder

This folder contains custom assets for your Chainlit application.

## Required Files

To customize your app's branding, add these files to this folder:

### 1. System Avatar (for AI responses)
- **File name**: `system-avatar.png`
- **Purpose**: This image will appear next to all system/AI responses
- **Recommended size**: 32x32px or 64x64px
- **Format**: PNG with transparent background works best

### 2. Main Logo (for header)
- **File names**: 
  - `logo_light.png` (for light mode)
  - `logo_dark.png` (for dark mode)
- **Purpose**: This appears in the header of your app
- **Recommended size**: 120x40px or similar aspect ratio
- **Format**: PNG with transparent background

### 3. Favicon (optional)
- **File name**: `favicon`
- **Purpose**: Browser tab icon
- **Recommended size**: 16x16px or 32x32px
- **Format**: ICO, PNG, or SVG

## How to Add Your Images

1. Place your image files in this `/public` folder
2. Make sure the filenames match exactly as specified above
3. Restart your Chainlit application
4. Clear your browser cache if you don't see changes immediately

## Alternative: Using URLs

If you prefer to host your images online, you can update the `config.toml` file to use URLs instead of local files:

```toml
default_avatar_file_url = "https://your-domain.com/path/to/your-avatar.png"
logo_file_url = "https://your-domain.com/path/to/your-logo.png"
```

## Current Configuration

Your app is currently configured to look for:
- System avatar: `/public/system-avatar.png`
- Main logo: `/public/logo_light.png` (and `/public/logo_dark.png` for dark mode)
