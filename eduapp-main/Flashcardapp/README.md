# SnapStudy - Interactive Learning App

A React Native mobile application built with Expo that gamifies learning through interactive study paths, AI-generated video summaries, and smart flashcards.

## Features

### 🗺️ Path (Home Screen)
- **Duolingo-style learning map** with winding path visualization
- Multiple levels with completion states (completed, active, locked)
- Animated mascot (🤖) guiding the current lesson
- Progress tracking with streak counter and points system
- SVG-based path rendering for smooth animations

### 📤 Upload Screen
- Upload PDF lecture slides to generate new study paths
- Alternative input methods:
  - Scan textbook pages with camera
  - Paste web links to study materials
- AI processing simulation with loading states
- Clean, intuitive drag-and-drop interface

### 🎥 Shorts (Video Summaries)
- YouTube Shorts-style vertical video player interface
- AI-generated video summaries of study material
- Interactive overlay UI with:
  - Like counter
  - Note-taking functionality
  - Share options
- Progress bar showing video position
- Highlighted key terms in descriptions

### 🃏 Flashcards
- Interactive flashcard system with flip animation
- Swipe controls (✓ for correct, ✗ for review)
- Progress tracking (current card / total cards)
- Smooth 3D flip animations using Animated API
- Timer display for study session tracking
- Multiple cards with auto-progression

### 👤 Profile Screen
- User statistics dashboard:
  - Day streak counter with fire icon
  - Total points earned
  - Completed lessons count
- Settings and account management
- Help & support access

## Tech Stack

- **Framework**: React Native with Expo
- **Navigation**: Expo Router (file-based routing)
- **UI Components**: 
  - @expo/vector-icons (Ionicons)
  - react-native-svg for path rendering
  - expo-linear-gradient for backgrounds
- **Animations**: React Native Animated API
- **Language**: TypeScript

## Installation

1. **Clone the repository**
   ```bash
   cd Flashcardapp
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Run on your device**
   - Scan the QR code with Expo Go (Android) or Camera app (iOS)
   - Or press `a` for Android emulator, `i` for iOS simulator

## Project Structure

```
Flashcardapp/
├── app/
│   ├── (tabs)/           # Tab-based screens
│   │   ├── index.tsx     # Path/Home screen
│   │   ├── cards.tsx     # Flashcards screen
│   │   ├── upload.tsx    # Upload screen
│   │   ├── shorts.tsx    # Video shorts screen
│   │   ├── profile.tsx   # Profile screen
│   │   └── _layout.tsx   # Tab navigation layout
│   └── _layout.tsx       # Root layout
├── components/           # Reusable UI components
├── constants/            # Theme and constants
└── assets/              # Images and static files
```

## Design Decisions (For Report/Proposal)

### User Experience (UX)
1. **Bottom Navigation Bar**: Thumb-friendly navigation placed at the bottom for easy one-handed use on mobile devices

2. **Card Metaphor**: Flashcards mimic real-world study tools, making the digital experience familiar and intuitive

3. **Vertical Video**: Uses TikTok/Instagram Reels-style vertical video format because students are already familiar with this interaction pattern

4. **Gamification**: Duolingo-inspired path with:
   - Visual progress indicators
   - Achievement system (streaks, points)
   - Unlockable content
   - Friendly mascot for engagement

5. **Progressive Disclosure**: Information revealed gradually:
   - Locked levels show future content
   - Flashcards hide answers until tapped
   - Upload screen offers multiple input methods

### Visual Design
- **Purple Primary Color** (#7c3aed): Conveys creativity and learning
- **Clear Visual Hierarchy**: Important actions use larger, more prominent buttons
- **Consistent Spacing**: 8px grid system for visual harmony
- **Shadow Depth**: Layered shadows create depth and interactivity cues
- **Color-Coded Feedback**:
  - Green (✓) for correct/positive actions
  - Red (✗) for review/negative actions
  - Purple for primary actions
  - Yellow for achievements

## Screenshots for Assignment

To capture screenshots for your assignment:

1. **Home/Path Screen**: Shows the learning map with mascot
2. **Upload Screen**: Click the center + button
3. **Shorts/Video Screen**: Click "Play" button on level 2 or "Shorts" tab
4. **Flashcards Screen**: Click "Cards" tab to show the flip card interface
5. **Profile Screen**: Click "Profile" tab for user stats

## Storyboard Flow

```
Upload PDF → Generate Path → Video Summary → Flashcards → Progress Saved
     ↓            ↓               ↓              ↓              ↓
[Upload UI]  [Path Map]      [Shorts UI]   [Card Flip]   [Profile Stats]
```

## Future Enhancements

- Real PDF processing with AI (OpenAI, Claude)
- Actual video generation from content
- Spaced repetition algorithm for flashcards
- Social features (friend leaderboards)
- Offline mode support
- Multiple subject support
- Custom path creation

## License

This project was created for educational purposes.

---

Built with ❤️ using React Native + Expo
