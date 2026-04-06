# Setup Guide - Poe AI Integration

## Prerequisites

1. Node.js and npm installed
2. Expo CLI installed globally: `npm install -g expo-cli`
3. Poe API key

## Getting Your Poe API Key

1. Visit [Poe API Keys](https://poe.com/api_key)
2. Sign in with your Poe account
3. Create a new API key
4. Copy the generated API key

## Configuration Steps

### 1. Install Dependencies

```bash
cd Flashcardapp
npm install
```

### 2. Set Up Environment Variables

The `.env` file already exists. Open it and replace the placeholder:

```bash
# Edit .env file
EXPO_PUBLIC_GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
```

**Important**: Never commit your `.env` file to git. It's already in `.gitignore`.

### 3. Start the Development Server

```bash
npm start
```

Or with cache cleared:
```bash
npx expo start --clear
```

### 4. Open the App

- **Web**: Press `w` in the terminal
- **Android**: Press `a` (requires emulator or device)
- **iOS**: Press `i` (requires simulator, macOS only)
- **Mobile Device**: Scan the QR code with Expo Go app

## How the AI Integration Works

### PDF Processing Flow

1. **Upload**: User selects a PDF file
2. **Extract**: Gemini AI extracts text from the PDF
3. **Generate**: AI creates three types of content:
   - Flashcards (10-15 cards with terms and definitions)
   - Study Path (5-7 progressive learning levels)
   - Video Summaries (3-5 short educational scripts)
4. **Save**: Content is stored in AsyncStorage
5. **Display**: Content appears in Cards, Home, and Shorts tabs

### API Usage

All AI functions are in `utils/gemini.ts`:

- `extractPDFContent()` - Extracts text from PDF using Gemini's vision capabilities
- `generateFlashcards()` - Creates flashcards from content
- `generateStudyPath()` - Designs learning progression
- `generateShortSummaries()` - Generates video scripts
- `processPDFAndGenerateContent()` - Main function that orchestrates everything

### API Key Requirements

- Gemini 1.5 Flash model is used (free tier available)
- The API key must support PDF/image processing
- Free tier includes generous quotas for testing

## Testing the Integration

1. Go to the Upload tab
2. Click "Tap to Upload PDF"
3. Select any educational PDF file (< 25MB)
4. Click "Generate with AI"
5. Wait for processing (shows progress updates)
6. Check success alert with generated content count
7. Navigate to other tabs to see the generated materials

## Troubleshooting

### "API Key Required" Error
- Check that `.env` file exists in the project root
- Verify `EXPO_PUBLIC_GEMINI_API_KEY` is set correctly
- Restart the development server after changing `.env`

### "Failed to extract PDF content" Error
- Ensure your API key has PDF processing enabled
- Try a smaller PDF file
- Check your internet connection
- Verify API quota hasn't been exceeded

### PDF Upload Not Working on Web
- This is expected - native file picker is used for web
- File should appear after selection
- Check browser console for errors

### Generated Content Not Appearing
- Content is stored in AsyncStorage
- Clear app data if testing repeatedly
- Check console logs for errors during generation

## File Structure

```
Flashcardapp/
├── .env                          # Your API key (not in git)
├── .env.example                  # Template for .env
├── utils/
│   └── gemini.ts                 # AI processing functions
├── app/
│   └── (tabs)/
│       ├── upload.tsx            # PDF upload & AI processing
│       ├── cards.tsx             # Display generated flashcards
│       ├── index.tsx             # Display generated study path
│       └── shorts.tsx            # Display generated summaries
└── README.md                     # Project overview
```

## Environment Variables Reference

```bash
# Required
EXPO_PUBLIC_GEMINI_API_KEY=your_key_here

# Optional
EXPO_PUBLIC_APP_NAME=SnapStudy
EXPO_PUBLIC_API_URL=http://localhost:3000
```

The `EXPO_PUBLIC_` prefix makes the variable accessible in the client-side code.

## Next Steps

After successful setup:

1. Upload a sample PDF to test the integration
2. Explore the generated content in different tabs
3. Customize the AI prompts in `utils/gemini.ts` for your use case
4. Integrate the generated content with existing components
5. Add data persistence beyond AsyncStorage if needed

## Support

If you encounter issues:
- Check the console/terminal for error messages
- Verify all dependencies are installed
- Ensure your API key is valid and has quota remaining
- Restart the development server

Happy coding! 🎓✨
