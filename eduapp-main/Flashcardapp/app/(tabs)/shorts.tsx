import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Dimensions,
  Alert,
  Share,
  ActivityIndicator,
} from 'react-native';
import styles from '@/constants/tab-styles/shorts_styles';
import { VideoView, useVideoPlayer } from 'expo-video';
import { useEventListener } from 'expo';
import { Ionicons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Sharing from 'expo-sharing';
import { File, Paths } from 'expo-file-system';
import { ShortVideo, buildCourseSummaryText, CourseForSummary } from '../../utils/gemini';

const { width, height } = Dimensions.get('window');

export default function ShortsScreen() {
  const router = useRouter();
  const [shortVideos, setShortVideos] = useState<ShortVideo[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [isLiked, setIsLiked] = useState(false);
  const [likeCount, setLikeCount] = useState(1200);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [useFallback, setUseFallback] = useState(false);
  const [screenKey, setScreenKey] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(0.5); // Speed control: 0.5x default
  
  // Track video operation state to prevent race conditions
  const isReplacingRef = useRef(false);
  const isPlayingRef = useRef(false);

  // Setup video player with looping - use RAMPVIDEO first, fallback to BigBuckBunny if fails
  const primarySource = require('../../assets/RAMPVIDEO.mp4');
  const fallbackSource = 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4';
  const videoSource = useFallback ? fallbackSource : primarySource;
  
  const player = useVideoPlayer(videoSource, (player) => {
    player.loop = true;
    player.muted = true;
    player.timeUpdateEventInterval = 0.1;
  });

  // Reset and replay video when shortVideos or screenKey changes
  useEffect(() => {
    if (!player || shortVideos.length === 0) return;
    
    let isMounted = true;

    const resetVideo = async () => {
      try {
        // Reset guard so focus events never leave us stuck
        isReplacingRef.current = false;
        isPlayingRef.current = false;

        // Seek to start
        try { player.currentTime = 0; } catch (e) {}

        // Small delay to let the player stabilise
        await new Promise(resolve => setTimeout(resolve, 100));
        if (!isMounted) return;

        player.loop = true;
        player.muted = true;

        // If using the fallback source, replace; otherwise just seek
        if (useFallback) {
          try {
            await player.replaceAsync(fallbackSource);
            await new Promise(resolve => setTimeout(resolve, 100));
          } catch (e) {
            console.log('Fallback replace error:', e);
          }
        }

        if (!isMounted) return;

        try {
          player.play();
        } catch (playError) {
          console.log('Play error (safe):', playError);
        }
      } catch (error) {
        console.error('Error in resetVideo:', error);
      }
    };

    resetVideo();

    return () => {
      isMounted = false;
    };
  }, [shortVideos, screenKey, player, useFallback]);

  useEventListener(player, 'timeUpdate', ({ currentTime }) => {
    setCurrentTime(currentTime);
  });

  // Detect video errors
  useEventListener(player, 'statusChange', ({ status, error }) => {
    if (error) {
      console.error('Video player error:', error);
      setVideoError(error.message);
      // Switch to fallback if primary source fails
      if (!useFallback) {
        setUseFallback(true);
      }
    }
  });

  // Generate timed content array (2 seconds each, adjusted by speed)
  const timedContent = useMemo(() => {
    if (!shortVideos[currentIndex]) return [];
    const video = shortVideos[currentIndex];
    const parts = [];
    const baseInterval = 2 / playbackSpeed; // Adjust timing based on speed
    
    // Add title first (0-baseInterval seconds)
    parts.push({ time: 0, text: video.title, type: 'title' });
    
    // Add description (baseInterval - 2*baseInterval)
    parts.push({ time: baseInterval, text: video.description, type: 'description' });
    
    // Add key points (2*baseInterval + each)
    if (video.keyPoints && video.keyPoints.length > 0) {
      video.keyPoints.forEach((point, index) => {
        parts.push({ time: baseInterval * 2 + (index * baseInterval), text: point, type: 'keypoint' });
      });
    }
    
    return parts;
  }, [shortVideos, currentIndex, playbackSpeed]);

  // Get current content based on time
  const currentContent = useMemo(() => {
    if (!timedContent || timedContent.length === 0) return null;
    let activeContent = timedContent[0];
    for (let i = timedContent.length - 1; i >= 0; i--) {
      if (currentTime >= timedContent[i].time) {
        activeContent = timedContent[i];
        break;
      }
    }
    return activeContent;
  }, [timedContent, currentTime]);

  const loadShortVideos = useCallback(async () => {
    try {
      setLoading(true);
      // Reset all state before loading new videos
      setCurrentTime(0);
      setCurrentIndex(0);
      setVideoError(null);
      setUseFallback(false);
      setScreenKey(prev => prev + 1); // Force component remount

      const selectedCourseJson = await AsyncStorage.getItem('selectedCourse');
      if (selectedCourseJson) {
        try {
          const selectedCourse = JSON.parse(selectedCourseJson);
          if (selectedCourse?.shortVideos?.length > 0) {
            setShortVideos(selectedCourse.shortVideos);
            setLoading(false);
            return;
          }
        } catch (parseError) {
          console.error('Error parsing selectedCourse:', parseError);
        }
      }

      const data = await AsyncStorage.getItem('studyData');
      if (data) {
        try {
          const parsedData = JSON.parse(data);
          if (parsedData.courses && parsedData.courses.length > 0) {
            const allShortVideos = parsedData.courses.flatMap((course: any) => course.shortVideos || []);
            if (allShortVideos.length > 0) {
              setShortVideos(allShortVideos);
            }
          }
        } catch (parseError) {
          console.error('Error parsing studyData:', parseError);
        }
      }
    } catch (error) {
      console.error('Error loading short videos:', error);
      setVideoError('Failed to load videos. Please try again.');
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      console.log('Shorts screen focused - reloading videos');
      // Always clear stuck refs on focus
      isReplacingRef.current = false;
      isPlayingRef.current = false;
      setCurrentTime(0);
      loadShortVideos();
      
      return () => {
        // Cleanup: stop operations when leaving the screen
        console.log('Shorts screen blurred');
        // Don't pause if we're in the middle of a replacement operation
        if (player && !isReplacingRef.current) {
          try {
            player.pause();
          } catch (e) {
            // Silently ignore pause errors
          }
        }
      };
    }, [player, loadShortVideos])
  );

  if (loading) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator size="large" color="#ffffff" />
        <Text style={{ marginTop: 16, color: '#ffffff' }}>Loading videos...</Text>
      </View>
    );
  }

  if (shortVideos.length === 0) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center', padding: 20 }]}>
        <Ionicons name="videocam-outline" size={64} color="#9ca3af" />
        <Text style={{ fontSize: 20, fontWeight: 'bold', color: '#ffffff', marginTop: 16 }}>No Videos Yet</Text>
        <Text style={{ fontSize: 14, color: '#9ca3af', marginTop: 8, textAlign: 'center' }}>Upload a PDF to generate video summaries</Text>
      </View>
    );
  }

  const currentVideo = shortVideos[currentIndex];

  const handleLike = () => {
    if (isLiked) {
      setIsLiked(false);
      setLikeCount(likeCount - 1);
    } else {
      setIsLiked(true);
      setLikeCount(likeCount + 1);
    }
  };

  const handleDownload = async () => {
    try {
      // Build the summary from the selected course data stored in AsyncStorage
      const selectedCourseJson = await AsyncStorage.getItem('selectedCourse');
      if (!selectedCourseJson) {
        Alert.alert('No Course Selected', 'Select a course on the home screen first.');
        return;
      }

      const selectedCourse = JSON.parse(selectedCourseJson) as CourseForSummary & { title: string };
      const summaryText = buildCourseSummaryText(selectedCourse);

      // Write to a temp file in the documents directory
      const safeTitle = (selectedCourse.title || 'study-summary')
        .replace(/[^a-z0-9]/gi, '_')
        .toLowerCase()
        .slice(0, 40);
      const summaryFile = new File(Paths.document, `${safeTitle}_summary.txt`);

      if (await summaryFile.exists) {
        await summaryFile.delete();
      }
      await summaryFile.create();
      await summaryFile.write(summaryText);

      const canShare = await Sharing.isAvailableAsync();
      if (canShare) {
        await Sharing.shareAsync(summaryFile.uri, {
          mimeType: 'text/plain',
          dialogTitle: 'Save or share study summary',
          UTI: 'public.plain-text',
        });
      } else {
        const { Linking } = await import('react-native');
        await Linking.openURL(summaryFile.uri);
      }
    } catch (error) {
      console.error('Download error:', error);
      Alert.alert('Download Failed', 'Could not generate the study summary. Please try again.');
    }
  };

  const handleShare = async () => {
    try {
      await Share.share({
        message: 'Check out this amazing study video on SnapStudy! ??',
        title: 'SnapStudy - Cell Structure',
      });
    } catch (error) {
      Alert.alert('Error', 'Could not share this video.');
    }
  };

  return (
    <View style={styles.container}>
      {/* Video Background - Looping */}
      <VideoView
        key={screenKey}
        style={styles.videoBackground}
        player={player}
        nativeControls={false}
        contentFit="cover"
      />

      {/* Dark Overlay for better text readability */}
      <View style={styles.gradientOverlay} />

      {/* Top Controls */}
      <View style={styles.topControls}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
          activeOpacity={0.7}
        >
          <Ionicons name="chevron-back" size={24} color="#ffffff" />
        </TouchableOpacity>
        <View style={styles.aiLabelContainer}>
          <View style={styles.aiLabel}>
            <Text style={styles.aiLabelText}>AI Summary</Text>
          </View>
          
          {/* Speed Control Buttons */}
          <View style={styles.speedControlContainer}>
            <TouchableOpacity 
              style={[styles.speedButton, playbackSpeed === 0.5 && styles.speedButtonActive]}
              onPress={() => setPlaybackSpeed(0.5)}
              activeOpacity={0.7}
            >
              <Text style={[styles.speedButtonText, playbackSpeed === 0.5 && styles.speedButtonTextActive]}>0.5x</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.speedButton, playbackSpeed === 1 && styles.speedButtonActive]}
              onPress={() => setPlaybackSpeed(1)}
              activeOpacity={0.7}
            >
              <Text style={[styles.speedButtonText, playbackSpeed === 1 && styles.speedButtonTextActive]}>1x</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.speedButton, playbackSpeed === 1.5 && styles.speedButtonActive]}
              onPress={() => setPlaybackSpeed(1.5)}
              activeOpacity={0.7}
            >
              <Text style={[styles.speedButtonText, playbackSpeed === 1.5 && styles.speedButtonTextActive]}>1.5x</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.speedButton, playbackSpeed === 2 && styles.speedButtonActive]}
              onPress={() => setPlaybackSpeed(2)}
              activeOpacity={0.7}
            >
              <Text style={[styles.speedButtonText, playbackSpeed === 2 && styles.speedButtonTextActive]}>2x</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* Center Content - Large Subtitles (Like Reels) */}
      <View style={styles.centerContent}>
        {currentContent?.type === 'title' && (
          <View style={styles.titleSection}>
            <View style={styles.titleBadge}>
              <Text style={styles.badgeText}>?? TITLE</Text>
            </View>
            <Text style={styles.videoTitle}>
              {currentContent.text}
            </Text>
          </View>
        )}
        
        {currentContent?.type === 'description' && (
          <View style={styles.descriptionSection}>
            <View style={styles.descriptionBadge}>
              <Text style={styles.badgeText}>?? OVERVIEW</Text>
            </View>
            <Text style={styles.description}>
              {currentContent.text}
            </Text>
          </View>
        )}
        
        {currentContent?.type === 'keypoint' && (
          <View style={styles.keypointSection}>
            <View style={styles.keypointBadge}>
              <Text style={styles.badgeText}>�?KEY POINT</Text>
            </View>
            <Text style={styles.keyPoint}>
              {currentContent.text}
            </Text>
          </View>
        )}
        
        {!currentContent && (
          <Text style={styles.loadingText}>Loading...</Text>
        )}
      </View>

      {/* Right Side Actions */}
      <View style={styles.rightActions}>
        <TouchableOpacity 
          style={styles.actionButton} 
          activeOpacity={0.7}
          onPress={handleLike}
        >
          <View style={styles.actionIcon}>
            <Ionicons 
              name={isLiked ? "heart" : "heart-outline"} 
              size={28} 
              color={isLiked ? "#ef4444" : "#ffffff"} 
            />
          </View>
          <Text style={styles.actionText}>{likeCount >= 1000 ? `${(likeCount / 1000).toFixed(1)}k` : likeCount}</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.actionButton} 
          activeOpacity={0.7}
          onPress={handleDownload}
        >
          <View style={styles.actionIcon}>
            <Ionicons name="download-outline" size={28} color="#ffffff" />
          </View>
          <Text style={styles.actionText}>Download</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.actionButton} 
          activeOpacity={0.7}
          onPress={handleShare}
        >
          <View style={styles.actionIcon}>
            <Ionicons name="share-social" size={28} color="#ffffff" />
          </View>
          <Text style={styles.actionText}>Share</Text>
        </TouchableOpacity>
      </View>

      {/* Progress Bar */}
      <View style={styles.progressBarContainer}>
        <View style={[styles.progressBar, { width: `${Math.min((currentTime / (player.duration || 1)) * 100, 100)}%` }]} />
      </View>
    </View>
  );
}

// Styles are in ./styles/shorts_styles.ts
