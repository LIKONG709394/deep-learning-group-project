import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Animated,
} from 'react-native';
import Svg, { Circle, Defs, RadialGradient, Stop, Line } from 'react-native-svg';
import * as Speech from 'expo-speech';
import styles from '@/constants/tab-styles/player_styles';
import { Ionicons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

// ── Space CD — colours mirror the home page ────────────────────────────────
const SpaceCD = () => (
  <Svg width={240} height={240} viewBox="0 0 240 240">
    <Defs>
      <RadialGradient id="bgGrad" cx="50%" cy="50%" r="50%">
        <Stop offset="0%" stopColor="#0F172A" />
        <Stop offset="60%" stopColor="#0d1b3e" />
        <Stop offset="100%" stopColor="#060D1A" />
      </RadialGradient>
      {/* Centre label — purple-to-blue, matches home's #7c3aed */}
      <RadialGradient id="labelGrad" cx="50%" cy="50%" r="50%">
        <Stop offset="0%" stopColor="#7C3AED" />
        <Stop offset="50%" stopColor="#3B82F6" />
        <Stop offset="100%" stopColor="#1E3A8A" />
      </RadialGradient>
      {/* Sun halo — golden, matches home sun */}
      <RadialGradient id="sunGlow" cx="50%" cy="50%" r="50%">
        <Stop offset="0%" stopColor="#FCD34D" stopOpacity="0.9" />
        <Stop offset="100%" stopColor="#F59E0B" stopOpacity="0" />
      </RadialGradient>
      <RadialGradient id="holeGlow" cx="50%" cy="50%" r="50%">
        <Stop offset="0%" stopColor="#7C3AED" stopOpacity="0.8" />
        <Stop offset="100%" stopColor="#0F1419" stopOpacity="0" />
      </RadialGradient>
    </Defs>
    {/* Main disc */}
    <Circle cx={120} cy={120} r={118} fill="url(#bgGrad)" />
    {/* Groove rings */}
    <Circle cx={120} cy={120} r={112} fill="none" stroke="#1E3A5F" strokeWidth={1.5} opacity={0.6} />
    <Circle cx={120} cy={120} r={104} fill="none" stroke="#164E63" strokeWidth={1.5} opacity={0.55} />
    <Circle cx={120} cy={120} r={96}  fill="none" stroke="#1E3A5F" strokeWidth={1.5} opacity={0.6} />
    <Circle cx={120} cy={120} r={88}  fill="none" stroke="#2D1B6E" strokeWidth={1.5} opacity={0.5} />
    <Circle cx={120} cy={120} r={80}  fill="none" stroke="#164E63" strokeWidth={1.5} opacity={0.55} />
    <Circle cx={120} cy={120} r={72}  fill="none" stroke="#1E3A5F" strokeWidth={1.5} opacity={0.6} />
    {/* Nebula glow rings — blue & teal matching home */}
    <Circle cx={120} cy={120} r={102} fill="none" stroke="#3B82F6" strokeWidth={5} opacity={0.10} />
    <Circle cx={120} cy={120} r={86}  fill="none" stroke="#06B6D4" strokeWidth={4} opacity={0.15} />
    <Circle cx={120} cy={120} r={70}  fill="none" stroke="#7C3AED" strokeWidth={4} opacity={0.20} />
    {/* Stars */}
    <Circle cx={28}  cy={55}  r={1.8} fill="white"   opacity={0.9} />
    <Circle cx={198} cy={72}  r={1.5} fill="white"   opacity={0.8} />
    <Circle cx={42}  cy={170} r={2.0} fill="white"   opacity={0.85} />
    <Circle cx={186} cy={175} r={1.5} fill="white"   opacity={0.75} />
    <Circle cx={80}  cy={22}  r={1.8} fill="white"   opacity={0.9} />
    <Circle cx={142} cy={218} r={1.5} fill="white"   opacity={0.8} />
    <Circle cx={16}  cy={115} r={2.0} fill="white"   opacity={0.85} />
    <Circle cx={222} cy={120} r={1.8} fill="white"   opacity={0.9} />
    <Circle cx={62}  cy={70}  r={1.5} fill="#A78BFA" opacity={0.9} />
    <Circle cx={168} cy={55}  r={1.8} fill="#60A5FA" opacity={0.85} />
    <Circle cx={30}  cy={145} r={1.5} fill="white"   opacity={0.75} />
    <Circle cx={203} cy={158} r={2.0} fill="#A78BFA" opacity={0.8} />
    {/* Planet 1 — teal/cyan, home Earth (#06B6D4) */}
    <Circle cx={62}  cy={88}  r={11}  fill="#0E7490" opacity={0.9} />
    <Circle cx={62}  cy={88}  r={8}   fill="#06B6D4" opacity={1.0} />
    <Circle cx={59}  cy={85}  r={3}   fill="#67E8F9" opacity={0.6} />
    <Circle cx={62}  cy={88}  r={16}  fill="none" stroke="#06B6D4" strokeWidth={1.5} opacity={0.45} />
    {/* Planet 2 — golden/yellow, home Sun (#FCD34D) */}
    <Circle cx={174} cy={158} r={9}   fill="#B45309" opacity={0.9} />
    <Circle cx={174} cy={158} r={7}   fill="#F59E0B" opacity={1.0} />
    <Circle cx={171} cy={155} r={2.5} fill="#FDE68A" opacity={0.7} />
    <Circle cx={174} cy={158} r={14}  fill="none" stroke="#FCD34D" strokeWidth={1.5} opacity={0.40} />
    <Circle cx={174} cy={158} r={20}  fill="url(#sunGlow)" opacity={0.3} />
    {/* Centre label */}
    <Circle cx={120} cy={120} r={56}  fill="url(#labelGrad)" />
    <Circle cx={120} cy={120} r={53}  fill="none" stroke="#A78BFA" strokeWidth={1.5} opacity={0.5} />
    <Circle cx={120} cy={120} r={46}  fill="none" stroke="#60A5FA" strokeWidth={1.0} opacity={0.4} />
    <Circle cx={120} cy={120} r={39}  fill="none" stroke="#C4B5FD" strokeWidth={1.5} opacity={0.35} />
    {/* Shooting streaks */}
    <Line x1={103} y1={107} x2={134} y2={98}  stroke="white"   strokeWidth={1.0} opacity={0.6} />
    <Line x1={94}  y1={133} x2={116} y2={127} stroke="#A78BFA" strokeWidth={0.8} opacity={0.5} />
    {/* Hole */}
    <Circle cx={120} cy={120} r={20}  fill="url(#holeGlow)" />
    <Circle cx={120} cy={120} r={13}  fill="#0A0F18" />
    <Circle cx={120} cy={120} r={13}  fill="none" stroke="#7C3AED" strokeWidth={2} opacity={0.9} />
    <Circle cx={120} cy={120} r={8}   fill="#060D1A" />
  </Svg>
);

interface Course {
  id: number;
  title: string;
  completed: boolean;
  locked: boolean;
  flashcards: any[];
  studyPath: any[];
  shortVideos: any[];
}

export default function PlayerScreen() {
  const router = useRouter();
  const [course, setCourse] = useState<Course | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isShuffle, setIsShuffle] = useState(false);
  const [loopMode, setLoopMode] = useState<'off' | 'all' | 'one'>('off');
  const [currentPage, setCurrentPage] = useState(0);
  const [speechKey, setSpeechKey] = useState(0);
  const [rotation] = useState(new Animated.Value(0));
  const rotationAnimRef = useRef<Animated.CompositeAnimation | null>(null);
  const [sunRotate] = useState(new Animated.Value(0));
  const [earthRotate] = useState(new Animated.Value(0));
  const [starsOpacity] = useState(new Animated.Value(0.3));

  // Background animations — same as home page
  useEffect(() => {
    Animated.loop(
      Animated.timing(sunRotate, { toValue: 360, duration: 20000, useNativeDriver: true })
    ).start();
    Animated.loop(
      Animated.timing(earthRotate, { toValue: 360, duration: 15000, useNativeDriver: true })
    ).start();
    Animated.loop(
      Animated.sequence([
        Animated.timing(starsOpacity, { toValue: 1, duration: 1500, useNativeDriver: true }),
        Animated.timing(starsOpacity, { toValue: 0.3, duration: 1500, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadCourse();
      setIsPlaying(false);
      return () => {
        Speech.stop();
        if (rotationAnimRef.current) {
          rotationAnimRef.current.stop();
          rotationAnimRef.current = null;
        }
      };
    }, [])
  );

  // CD rotation
  useEffect(() => {
    if (isPlaying) {
      rotationAnimRef.current = Animated.loop(
        Animated.timing(rotation, {
          toValue: 1,
          duration: 3000,
          useNativeDriver: true,
        })
      );
      rotationAnimRef.current.start();
    } else {
      if (rotationAnimRef.current) {
        rotationAnimRef.current.stop();
        rotationAnimRef.current = null;
      }
    }
  }, [isPlaying]);

  // Text-to-speech playback
  useEffect(() => {
    if (!isPlaying || !course) {
      Speech.stop();
      return;
    }
    const content = course.studyPath || [];
    const current = content[currentPage];
    if (!current) return;

    const textToSpeak = [
      current.title || `Section ${currentPage + 1}`,
      current.content || current.description || '',
    ]
      .filter(Boolean)
      .join('. ');

    Speech.speak(textToSpeak, {
      language: 'en-US',
      rate: 0.85,
      pitch: 1.0,
      onDone: () => {
        if (loopMode === 'one') {
          setSpeechKey((k) => k + 1);
        } else {
          const next = currentPage + 1;
          if (next >= content.length) {
            if (loopMode === 'all') setCurrentPage(0);
            else setIsPlaying(false);
          } else {
            setCurrentPage(next);
          }
        }
      },
      onError: () => { /* silently skip */ },
    });

    return () => {
      Speech.stop();
    };
  }, [isPlaying, currentPage, course, loopMode, speechKey]);

  const loadCourse = async () => {
    try {
      const data = await AsyncStorage.getItem('selectedCourse');
      if (data) {
        const parsed = JSON.parse(data);
        setCourse(parsed);
        setCurrentPage(0);
      } else {
        setCourse(null);
      }
    } catch (error) {
      console.error('Error loading course:', error);
    }
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      Speech.stop();
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
    }
  };

  const handleNext = () => {
    Speech.stop();
    const content = course?.studyPath || [];
    if (isShuffle) {
      setCurrentPage(Math.floor(Math.random() * content.length));
    } else {
      setCurrentPage((prev) => (prev + 1) % content.length);
    }
  };

  const handlePrevious = () => {
    Speech.stop();
    const content = course?.studyPath || [];
    setCurrentPage((prev) => (prev - 1 + content.length) % content.length);
  };

  const toggleLoop = () => {
    if (loopMode === 'off') setLoopMode('all');
    else if (loopMode === 'all') setLoopMode('one');
    else setLoopMode('off');
  };

  const toggleShuffle = () => {
    setIsShuffle(!isShuffle);
  };

  const SpaceBg = () => (
    <View style={styles.backgroundContainer}>
      <View style={styles.spaceGradient} />
      <Animated.View style={[styles.sunContainer, { transform: [{ rotate: sunRotate.interpolate({ inputRange: [0, 360], outputRange: ['0deg', '360deg'] }) }] }]}>
        <View style={styles.sun}><View style={styles.sunGlow} /></View>
      </Animated.View>
      <Animated.View style={[styles.earthContainer, { transform: [{ rotate: earthRotate.interpolate({ inputRange: [0, 360], outputRange: ['0deg', '360deg'] }) }] }]}>
        <View style={styles.earth}><View style={styles.earthDetail} /></View>
      </Animated.View>
      <Animated.View style={[styles.starsContainer, { opacity: starsOpacity }]}>
        <View style={[styles.star, { top: '8%',  left: '15%' }]} />
        <View style={[styles.star, { top: '18%', right: '20%' }]} />
        <View style={[styles.star, { top: '35%', left: '8%' }]} />
        <View style={[styles.star, { top: '50%', right: '12%' }]} />
        <View style={[styles.star, { bottom: '35%', left: '22%' }]} />
        <View style={[styles.star, { bottom: '20%', right: '18%' }]} />
        <View style={[styles.star, { bottom: '40%', right: '6%' }]} />
        <View style={[styles.star, { top: '65%', left: '10%' }]} />
      </Animated.View>
      <View style={styles.particlesContainer}>
        <View style={[styles.particle, { top: '20%', left: '25%' }]} />
        <View style={[styles.particle, { top: '45%', right: '30%' }]} />
        <View style={[styles.particle, { top: '65%', left: '35%' }]} />
        <View style={[styles.particle, { bottom: '25%', right: '25%' }]} />
      </View>
    </View>
  );

  if (!course) {
    return (
      <View style={styles.container}>
        <SpaceBg />
        <View style={styles.header}>
          <View style={{ width: 28 }} />
          <Text style={styles.headerText}>MUSIC PLAYER</Text>
          <View style={{ width: 28 }} />
        </View>
        
        <View style={styles.emptyStateContainer}>
          <View style={styles.emptyAlbumArt}>
            <Ionicons name="musical-notes" size={80} color="#3B82F6" />
          </View>
          <Text style={styles.emptyTitle}>No Song Playing</Text>
          <Text style={styles.emptyDescription}>
            Upload a PDF from the Upload tab to start studying with your personal music player!
          </Text>
          
          <TouchableOpacity 
            style={styles.uploadButton}
            onPress={() => router.push('/upload')}
          >
            <Ionicons name="cloud-upload" size={24} color="#FFFFFF" />
            <Text style={styles.uploadButtonText}>Upload PDF</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const content = course?.studyPath || [];
  const currentContent = content[currentPage];

  const rotate = rotation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return (
    <View style={styles.container}>
      <SpaceBg />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="chevron-down" size={28} color="#FFFFFF" />
        </TouchableOpacity>
        <Text style={styles.headerText}>NOW STUDYING</Text>
        <View style={{ width: 28 }} />
      </View>

      {/* Space CD */}
      <View style={styles.albumContainer}>
        <Animated.View
          style={[
            styles.albumArt,
            {
              transform: [{ rotate: isPlaying ? rotate : '0deg' }],
            },
          ]}
        >
          <SpaceCD />
        </Animated.View>
      </View>

      {/* Course Info */}
      <View style={styles.infoContainer}>
        <Text style={styles.courseTitle}>{course.title}</Text>
        <Text style={styles.contentCount}>
          {currentPage + 1} / {content.length} sections
        </Text>
      </View>

      {/* Progress Bar */}
      <View style={styles.progressContainer}>
        <View
          style={[
            styles.progressBar,
            { width: ((currentPage + 1) / content.length) * (width - 48) },
          ]}
        />
      </View>

      {/* Controls */}
      <View style={styles.controlsContainer}>
        <TouchableOpacity onPress={toggleShuffle}>
          <Ionicons
            name="shuffle"
            size={24}
            color={isShuffle ? '#3B82F6' : '#6B7280'}
          />
        </TouchableOpacity>

        <TouchableOpacity onPress={handlePrevious}>
          <Ionicons name="play-skip-back" size={32} color="#FFFFFF" />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.playButton}
          onPress={handlePlayPause}
        >
          <Ionicons
            name={isPlaying ? 'pause' : 'play'}
            size={40}
            color="#FFFFFF"
          />
        </TouchableOpacity>

        <TouchableOpacity onPress={handleNext}>
          <Ionicons name="play-skip-forward" size={32} color="#FFFFFF" />
        </TouchableOpacity>

        <TouchableOpacity onPress={toggleLoop}>
          <Ionicons
            name={loopMode === 'one' ? 'repeat-one' : 'repeat'}
            size={24}
            color={loopMode !== 'off' ? '#3B82F6' : '#6B7280'}
          />
        </TouchableOpacity>
      </View>

      {/* Content / Lyrics */}
      <View style={styles.contentHeaderContainer}>
        <Text style={styles.contentHeader}>STUDY CONTENT</Text>
      </View>

      <ScrollView
        style={styles.lyricsContainer}
        contentContainerStyle={styles.lyricsContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.contentCard}>
          {currentContent ? (
            <>
              <Text style={styles.contentTitle}>
                {currentContent.title || `Section ${currentPage + 1}`}
              </Text>
              <Text style={styles.contentText}>
                {currentContent.content ||
                  currentContent.description ||
                  'No content available for this section'}
              </Text>
            </>
          ) : (
            <Text style={styles.contentText}>
              No content available. Please upload PDF content to get started.
            </Text>
          )}
        </View>
      </ScrollView>
    </View>
  );
}

// Styles are in ./styles/player_styles.ts
