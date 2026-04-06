# SnapStudy App - Quick Start Guide

## 🚀 Getting Started

### Prerequisites
- Node.js installed
- Expo Go app on your mobile device (iOS/Android)

### Installation & Running

1. Navigate to the project directory:
   ```bash
   cd /home/hou/Desktop/eduapp/Flashcardapp
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Scan the QR code with:
   - **Android**: Expo Go app
   - **iOS**: Camera app (opens in Expo Go)

4. Or run on web:
   ```bash
   Press 'w' in the terminal
   ```

---

## 📱 App Navigation

### Bottom Tab Bar (5 Tabs)

1. **Path** (Home) 🗺️
   - View your learning journey
   - Click on active level (purple button with 🤖) to start
   - See completed (yellow) and locked (gray) levels

2. **Cards** 🃏
   - Practice with flashcards
   - Tap card to flip and reveal definition
   - Swipe right (✓) if you know it
   - Swipe left (✗) if you need to review

3. **Upload** ➕ (Center button)
   - Upload PDF lecture notes
   - Scan textbook pages
   - Paste web links
   - Click "Generate Path ✨" to create new study material

4. **Shorts** 🎥
   - Watch AI-generated video summaries
   - Like, take notes, or share
   - Progress bar shows video position
   - Back button returns to Path

5. **Profile** 👤
   - View your stats (streak, points, completed)
   - Access settings
   - Get help & support

---

## 📸 Screenshots for Assignment

### For Your Report/Proposal:

1. **Main Page (Path View)**
   - Default screen when app opens
   - Shows: Learning map, mascot, levels, header with stats

2. **Input Screen (Upload)**
   - Click center **+** button in tab bar
   - Shows: Upload area, scan/link options, generate button

3. **Output 1 (Video/Shorts)**
   - Click **Play** button on purple level (Cell Structure)
   - OR click **Shorts** tab
   - Shows: Video player, overlay UI, progress bar

4. **Output 2 (Flashcards)**
   - Click **Cards** tab
   - Shows: Flashcard, flip animation, swipe controls
   - Take screenshot of both front and back of card

5. **Profile Stats**
   - Click **Profile** tab
   - Shows: User stats, menu items

---

## 🎨 Design Features to Mention in Report

### User Experience (UX)
- **Bottom Navigation**: Easy thumb access on mobile
- **Card Metaphor**: Familiar real-world study tool
- **Vertical Video**: TikTok/Instagram-style familiarity
- **Gamification**: Duolingo-inspired progression
- **Progressive Disclosure**: Information revealed gradually

### Visual Design
- **Purple Theme** (#7c3aed): Creative and educational
- **Color Coding**:
  - 🟢 Green: Correct/Positive
  - 🔴 Red: Review/Wrong
  - 🟣 Purple: Primary actions
  - 🟡 Yellow: Achievements/Completed
- **3D Effects**: Card shadows and depth
- **Animations**: Smooth transitions and feedback

### Technical Features
- **React Native + Expo**: Cross-platform mobile
- **SVG Path Rendering**: Smooth learning path
- **Animated API**: Flip and swipe animations
- **TypeScript**: Type-safe development
- **File-based Routing**: Expo Router for navigation

---

## 📊 App Flow Diagram (Storyboard)

```
┌─────────────┐
│   Upload    │ User uploads PDF/content
│   Screen    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Path     │ AI generates learning map
│    Screen   │ with levels
└──────┬──────┘
       │
       ↓ (Click level)
┌─────────────┐
│   Shorts    │ Watch AI video summary
│   Screen    │ (YouTube Shorts style)
└──────┬──────┘
       │
       ↓ (Swipe up or navigate)
┌─────────────┐
│   Cards     │ Practice with flashcards
│   Screen    │ (Swipe ✓ or ✗)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Profile    │ View progress & stats
│   Screen    │ (Streak, points, etc.)
└─────────────┘
```

---

## 🎯 Key App Features

### 1. Path/Home Screen
- **Winding Learning Path**: Visual representation of progress
- **Level States**: Completed (✓), Active (▶), Locked (🔒)
- **Mascot Guide**: Animated emoji showing current focus
- **Stats Header**: Streak 🔥 and Points 💎

### 2. Upload Screen
- **PDF Upload**: Drag-and-drop or tap to select
- **Camera Scan**: Capture textbook pages
- **Link Paste**: Import online content
- **AI Processing**: Simulated generation with loading state

### 3. Shorts/Video Screen
- **Vertical Video**: Full-screen immersive experience
- **Overlay Controls**: Like, comment, share buttons
- **AI Summary Badge**: Shows content is AI-generated
- **Key Terms Highlighted**: Important words emphasized
- **Progress Indicator**: Video timeline bar

### 4. Flashcards Screen
- **3D Flip Animation**: Tap to reveal definition
- **Swipe Controls**: Left (✗) = review, Right (✓) = know it
- **Progress Bar**: Shows completion percentage
- **Card Counter**: Current position (3/10)
- **Timer**: Study session duration

### 5. Profile Screen
- **Statistics Dashboard**: Visual cards showing metrics
- **Settings Access**: Account and app preferences
- **Help & Support**: Documentation and assistance

---

## 🛠️ Development Commands

```bash
# Start development server
npm start

# Run on Android emulator
npm run android

# Run on iOS simulator
npm run ios

# Run on web browser
npm run web

# Check for errors
npm run lint
```

---

## 📝 Notes for Assignment

### Justification Points:

1. **Mobile-First Design**
   - Bottom navigation for thumb reach
   - Large touch targets (48px minimum)
   - Swipe gestures for natural interaction

2. **Familiar UI Patterns**
   - Duolingo's path-based progression
   - TikTok's vertical video format
   - Physical flashcard metaphor

3. **Accessibility**
   - High contrast colors
   - Large, readable text
   - Clear visual feedback
   - Haptic feedback on interactions

4. **Engagement Features**
   - Gamification (points, streaks)
   - Progress visualization
   - Immediate feedback
   - Social sharing options

5. **AI Integration (Concept)**
   - PDF to study path generation
   - Video summary creation
   - Flashcard extraction
   - Personalized learning paths

---

## 🎓 Educational Value

- **Active Recall**: Flashcard system promotes memory retention
- **Spaced Repetition**: Track progress for optimal review timing
- **Multimodal Learning**: Video + text + interactive cards
- **Bite-sized Content**: Short videos and single-concept cards
- **Progress Tracking**: Visual feedback motivates continued learning

---

## 📦 Export for Submission

To create a demo video or screenshots:

1. **Run on device**: Use Expo Go for real device testing
2. **Screen recording**: 
   - iOS: Control Center → Screen Recording
   - Android: Quick Settings → Screen Record
3. **Screenshots**: Use device screenshot shortcuts
4. **Annotate**: Use image editor to highlight features

---

## ✅ Checklist for Report

- [ ] Screenshot of Path screen (main interface)
- [ ] Screenshot of Upload screen (input method)
- [ ] Screenshot of Shorts screen (output 1)
- [ ] Screenshot of Cards screen front (output 2a)
- [ ] Screenshot of Cards screen back (output 2b)
- [ ] Storyboard/flow diagram included
- [ ] Design justifications documented
- [ ] UX principles explained
- [ ] Color scheme rationale provided
- [ ] Technical stack described

---

**Good luck with your assignment! 🎉**
