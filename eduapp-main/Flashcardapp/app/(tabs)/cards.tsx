import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Animated,
  ActivityIndicator,
} from 'react-native';
import styles from '@/constants/tab-styles/cards_styles';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { FlashCard } from '../../utils/gemini';
import { useFocusEffect } from 'expo-router';

export default function CardsScreen() {
  const [flashcards, setFlashcards] = useState<FlashCard[]>([]);
  const [remainingCards, setRemainingCards] = useState<FlashCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [flipAnimation] = useState(new Animated.Value(0));
  const [slideAnimation] = useState(new Animated.Value(0));
  const [completedCount, setCompletedCount] = useState(0);

  useFocusEffect(
    useCallback(() => {
      loadFlashcards();
    }, [])
  );

  const loadFlashcards = async () => {
    try {
      setLoading(true);
      const data = await AsyncStorage.getItem('studyData');
      if (data) {
        const parsedData = JSON.parse(data);
        // Combine all flashcards from all courses
        if (parsedData.courses && parsedData.courses.length > 0) {
          const allFlashcards = parsedData.courses.flatMap((course: any) => course.flashcards || []);
          if (allFlashcards.length > 0) {
            setFlashcards(allFlashcards);
            setRemainingCards(allFlashcards); // Initialize remaining cards
            setCurrentIndex(0); // Reset to first card on new data
            setIsFlipped(false); // Reset flip state
            setCompletedCount(0); // Reset completed count
            flipAnimation.setValue(0);
          }
        }
      }
    } catch (error) {
      console.error('Error loading flashcards:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <ActivityIndicator size="large" color="#3B82F6" />
        <Text style={styles.loadingText}>Loading flashcards...</Text>
      </View>
    );
  }

  if (flashcards.length === 0) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <Ionicons name="document-text-outline" size={64} color="#9ca3af" />
        <Text style={styles.emptyTitle}>No Flashcards Yet</Text>
        <Text style={styles.emptyText}>Upload a PDF to generate flashcards</Text>
      </View>
    );
  }

  // If all cards completed
  if (remainingCards.length === 0) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <View style={styles.completionContainer}>
          <Text style={styles.completionEmoji}>??</Text>
          <Text style={styles.completionTitle}>All Cards Mastered!</Text>
          <Text style={styles.completionSubtitle}>
            You've completed {completedCount} cards successfully!
          </Text>
          <TouchableOpacity 
            style={styles.refreshMainButton}
            onPress={() => loadFlashcards()}
          >
            <Ionicons name="refresh" size={20} color="#FFFFFF" />
            <Text style={styles.refreshMainButtonText}>Review Again</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const currentCard = remainingCards[0];
  const progress = ((completedCount) / flashcards.length) * 100;

  const flipCard = () => {
    if (isFlipped) {
      Animated.timing(flipAnimation, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }).start();
    } else {
      Animated.timing(flipAnimation, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }).start();
    }
    setIsFlipped(!isFlipped);
  };

  // Mark card as learned (remove from queue)
  const markCorrect = () => {
    Animated.timing(slideAnimation, {
      toValue: 400,
      duration: 300,
      useNativeDriver: true,
    }).start(() => {
      slideAnimation.setValue(0);
      flipAnimation.setValue(0);
      setIsFlipped(false);
      
      // Remove current card from remaining
      const newRemaining = remainingCards.slice(1);
      setRemainingCards(newRemaining);
      setCompletedCount(completedCount + 1);
    });
  };

  // Send card to end of queue (need review)
  const markWrong = () => {
    Animated.timing(slideAnimation, {
      toValue: -400,
      duration: 300,
      useNativeDriver: true,
    }).start(() => {
      slideAnimation.setValue(0);
      flipAnimation.setValue(0);
      setIsFlipped(false);
      
      // Move current card to end of queue
      const currentCard = remainingCards[0];
      const newRemaining = [...remainingCards.slice(1), currentCard];
      setRemainingCards(newRemaining);
    });
  };

  const frontRotation = flipAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '180deg'],
  });

  const backRotation = flipAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: ['180deg', '360deg'],
  });

  return (
    <View style={styles.container}>
      {/* Progress Section */}
      <View style={styles.progressSection}>
        <View style={styles.headerWithRefresh}>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${progress}%` }]} />
          </View>
          <TouchableOpacity 
            style={styles.refreshButton}
            onPress={() => loadFlashcards()}
          >
            <Ionicons name="refresh" size={20} color="#3B82F6" />
          </TouchableOpacity>
        </View>
        <View style={styles.statsRow}>
          <Text style={styles.statsText}>
            Remaining: {remainingCards.length} / {flashcards.length} ??Completed: {completedCount}
          </Text>
          <View style={styles.timeContainer}>
            <Ionicons name="time-outline" size={14} color="#6b7280" />
            <Text style={styles.statsText}>2:00</Text>
          </View>
        </View>
      </View>

      {/* Card Container */}
      <View style={styles.cardSection}>
        <TouchableOpacity
          activeOpacity={1}
          onPress={flipCard}
          style={styles.cardTouchable}
        >
          <Animated.View
            style={[
              styles.cardContainer,
              {
                transform: [
                  { translateX: slideAnimation },
                ],
              },
            ]}
          >
            {/* Front of Card */}
            <Animated.View
              style={[
                styles.card,
                styles.cardFront,
                { transform: [{ perspective: 1000 }, { rotateY: frontRotation }] },
              ]}
            >
              <Text style={styles.cardLabel}>TERM</Text>
              <Text style={styles.cardTerm}>{currentCard.term}</Text>
              <Text style={styles.tapHint}>(Tap to reveal)</Text>
            </Animated.View>

            {/* Back of Card */}
            <Animated.View
              style={[
                styles.card,
                styles.cardBack,
                { transform: [{ perspective: 1000 }, { rotateY: backRotation }] },
              ]}
            >
              <Text style={styles.cardLabelBack}>DEFINITION</Text>
              <Text style={styles.cardDefinition}>{currentCard.definition}</Text>
            </Animated.View>
          </Animated.View>
        </TouchableOpacity>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.controlButton, styles.wrongButton]}
          onPress={markWrong}
          activeOpacity={0.7}
        >
          <Ionicons name="close" size={32} color="#ef4444" />
          <Text style={styles.buttonLabel}>Review</Text>
        </TouchableOpacity>

        <Text style={styles.swipeText}>TAP</Text>

        <TouchableOpacity
          style={[styles.controlButton, styles.correctButton]}
          onPress={markCorrect}
          activeOpacity={0.7}
        >
          <Ionicons name="checkmark" size={32} color="#10b981" />
          <Text style={styles.buttonLabel}>Done</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// Styles are in ./styles/cards_styles.ts
