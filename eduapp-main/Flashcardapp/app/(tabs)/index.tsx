import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  ActivityIndicator,
  Animated,
  Modal,
} from 'react-native';
import styles from '@/constants/tab-styles/index_styles';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Path, Circle, Ellipse, Rect, G } from 'react-native-svg';
import { useRouter, useFocusEffect } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { StudyLevel } from '../../utils/gemini';

const { width, height } = Dimensions.get('window');

interface Course {
  id: number;
  title: string;
  completed: boolean;
  locked: boolean;
  flashcards: any[];
  studyPath: StudyLevel[];
  shortVideos: any[];
}

export default function HomeScreen() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const [sunRotate] = useState(new Animated.Value(0));
  const [earthRotate] = useState(new Animated.Value(0));
  const [starsOpacity] = useState(new Animated.Value(0.3));
  const [streakVisible, setStreakVisible] = useState(false);
  const [diamondVisible, setDiamondVisible] = useState(false);
  const [diamonds, setDiamonds] = useState(420);
  const [badgesOwned, setBadgesOwned] = useState(new Set());
  const [badgesPurchased, setBadgesPurchased] = useState(0);

  // Badge data with prices
  const badgeData = [
    { id: 1, name: '🏆 Gold Badge', price: 75, description: '100 pts earned' },
    { id: 2, name: '⭐ Star Learner', price: 100, description: '7-day streak' },
    { id: 3, name: '🔥 On Fire', price: 50, description: '5 pts in a day' },
    { id: 4, name: '🎓 Scholar', price: 150, description: '5 courses done' },
    { id: 5, name: '📚 Bookworm', price: 75, description: 'Read 100 cards' },
    { id: 6, name: '🚀 Rocket Start', price: 50, description: '3 courses done' },
  ];

  const handleBuyBadge = (badgeId: number, price: number) => {
    if (diamonds >= price) {
      const newDiamonds = diamonds - price;
      const newBadgesOwned = new Set([...badgesOwned, badgeId]);
      const newBadgesPurchased = badgesPurchased + 1;
      
      setDiamonds(newDiamonds);
      setBadgesOwned(newBadgesOwned);
      setBadgesPurchased(newBadgesPurchased);
      
      // Save to AsyncStorage
      saveBadgeData(newDiamonds, Array.from(newBadgesOwned) as number[], newBadgesPurchased);
    } else {
      alert('Not enough diamonds! You need ' + (price - diamonds) + ' more diamonds.');
    }
  };

  const saveBadgeData = async (dmd: number, badges: number[], count: number) => {
    try {
      const data = {
        diamonds: dmd,
        badgesOwned: badges,
        badgesPurchased: count,
      };
      await AsyncStorage.setItem('badgeData', JSON.stringify(data));
    } catch (error) {
      console.error('Error saving badge data:', error);
    }
  };

  const loadBadgeData = async () => {
    try {
      const data = await AsyncStorage.getItem('badgeData');
      if (data) {
        const parsedData = JSON.parse(data);
        setDiamonds(parsedData.diamonds || 420);
        setBadgesOwned(new Set(parsedData.badgesOwned || []));
        setBadgesPurchased(parsedData.badgesPurchased || 0);
      }
    } catch (error) {
      console.error('Error loading badge data:', error);
    }
  };

  useEffect(() => {
    // Load badge data from AsyncStorage
    loadBadgeData();

    // Sun rotation
    Animated.loop(
      Animated.timing(sunRotate, {
        toValue: 360,
        duration: 20000,
        useNativeDriver: true,
      })
    ).start();

    // Earth rotation
    Animated.loop(
      Animated.timing(earthRotate, {
        toValue: 360,
        duration: 15000,
        useNativeDriver: true,
      })
    ).start();

    // Stars twinkling
    Animated.loop(
      Animated.sequence([
        Animated.timing(starsOpacity, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: true,
        }),
        Animated.timing(starsOpacity, {
          toValue: 0.3,
          duration: 1500,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadCourses();
      loadBadgeData();
    }, [])
  );

  const loadCourses = async () => {
    try {
      setLoading(true);
      const data = await AsyncStorage.getItem('studyData');
      if (data) {
        const parsedData = JSON.parse(data);
        if (parsedData.courses && parsedData.courses.length > 0) {
          setCourses(parsedData.courses);
          // Auto-select first active course if nothing is selected yet
          const existing = await AsyncStorage.getItem('selectedCourse');
          if (!existing) {
            const firstActive = parsedData.courses.find((c: any) => !c.locked) || parsedData.courses[0];
            await AsyncStorage.setItem('selectedCourse', JSON.stringify(firstActive));
          }
          setLoading(false);
          return;
        }
      }
      // No courses found ??clear selectedCourse so player/shorts also show empty state
      await AsyncStorage.removeItem('selectedCourse');
      setCourses([]);
    } catch (error) {
      console.error('Error loading courses:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.subjectIcon}>
            <Ionicons name="school" size={20} color="#7c3aed" />
          </View>
          <Text style={styles.subjectText}>My Courses</Text>
        </View>
        <View style={styles.headerRight}>
          <TouchableOpacity 
            style={styles.statItem}
            activeOpacity={0.7}
            onPress={() => setStreakVisible(true)}
          >
            <Ionicons name="flame" size={20} color="#f97316" />
            <Text style={styles.statText}>5</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={styles.statItem}
            activeOpacity={0.7}
            onPress={() => setDiamondVisible(true)}
          >
            <Ionicons name="diamond" size={20} color="#3b82f6" />
            <Text style={styles.statText}>{diamonds}</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Scrollable Map Area */}
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.mapContainer}
        showsVerticalScrollIndicator={false}
      >
        {/* Dynamic Space Background */}
        <View style={styles.backgroundContainer}>
          {/* Space gradient */}
          <View style={styles.spaceGradient} />
          
          {/* Animated Sun */}
          <Animated.View 
            style={[
              styles.sunContainer,
              {
                transform: [
                  {
                    rotate: sunRotate.interpolate({
                      inputRange: [0, 360],
                      outputRange: ['0deg', '360deg'],
                    }),
                  },
                ],
              }
            ]}
          >
            <View style={styles.sun}>
              <View style={styles.sunGlow} />
            </View>
          </Animated.View>

          {/* Animated Earth */}
          <Animated.View 
            style={[
              styles.earthContainer,
              {
                transform: [
                  {
                    rotate: earthRotate.interpolate({
                      inputRange: [0, 360],
                      outputRange: ['0deg', '360deg'],
                    }),
                  },
                ],
              }
            ]}
          >
            <View style={styles.earth}>
              <View style={styles.earthDetail} />
            </View>
          </Animated.View>

          {/* Twinkling Stars */}
          <Animated.View style={[styles.starsContainer, { opacity: starsOpacity }]}>
            <View style={[styles.star, { top: '10%', left: '15%' }]} />
            <View style={[styles.star, { top: '20%', right: '20%' }]} />
            <View style={[styles.star, { top: '35%', left: '10%' }]} />
            <View style={[styles.star, { top: '45%', right: '15%' }]} />
            <View style={[styles.star, { bottom: '30%', left: '20%' }]} />
            <View style={[styles.star, { bottom: '20%', right: '18%' }]} />
            <View style={[styles.star, { bottom: '40%', right: '5%' }]} />
            <View style={[styles.star, { top: '65%', left: '12%' }]} />
          </Animated.View>

          {/* Floating Dust Particles */}
          <View style={styles.particlesContainer}>
            <View style={[styles.particle, { top: '20%', left: '25%' }]} />
            <View style={[styles.particle, { top: '40%', right: '30%' }]} />
            <View style={[styles.particle, { top: '60%', left: '35%' }]} />
            <View style={[styles.particle, { bottom: '25%', right: '25%' }]} />
          </View>
        </View>

        {/* SVG Path - Snake Design */}
        <View style={styles.pathContainer}>
          <Svg height={Math.max(courses.length * 220 + 200, Math.floor(height * 1.5))} width={width}>
            {/* Static decorative S-path ??ALWAYS VISIBLE as background */}
            <Path
              d={`M${width/2} 40 C ${width*0.8} 80, ${width*0.8} 160, ${width/2} 200 C ${width*0.2} 240, ${width*0.2} 320, ${width/2} 360 C ${width*0.8} 400, ${width*0.8} 480, ${width/2} 520 C ${width*0.2} 560, ${width*0.2} 640, ${width/2} 680 C ${width*0.8} 720, ${width*0.8} 800, ${width/2} 840`}
              stroke="#3b82f6"
              strokeWidth="3"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.6"
            />
            <Path
              d={`M${width/2} 40 C ${width*0.8} 80, ${width*0.8} 160, ${width/2} 200 C ${width*0.2} 240, ${width*0.2} 320, ${width/2} 360 C ${width*0.8} 400, ${width*0.8} 480, ${width/2} 520 C ${width*0.2} 560, ${width*0.2} 640, ${width/2} 680 C ${width*0.8} 720, ${width*0.8} 800, ${width/2} 840`}
              stroke="rgba(129, 140, 248, 0.3)"
              strokeWidth="6"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.4"
            />

            {/* Background decorative circles ??course dependent */}
            {courses.map((_, i) => (
              <Circle
                key={`circle-${i}`}
                cx={width / 2}
                cy={80 + (i * 220)}
                r={120}
                fill="none"
                stroke="rgba(59, 130, 246, 0.1)"
                strokeWidth="1"
              />
            ))}
            

          </Svg>

          {courses.map((course, index) => {
            const isRight = index % 2 === 0;
            return (
            <View
              key={course.id}
              style={[
                styles.levelContainer,
                {
                  top: 50 + (index * 160),
                  left: isRight ? width * 0.65 : width * 0.1,
                }
              ]}
            >
              {course.completed && !course.locked && (
                <View style={[styles.levelButton, styles.completedLevel]}>
                  <Ionicons name="checkmark" size={32} color="#ffffff" />
                </View>
              )}
              {!course.completed && !course.locked && (
                <>
                  {index === 0 && (
                    <View style={styles.mascotContainer}>
                    </View>
                  )}
                  <TouchableOpacity
                    style={[styles.levelButton, styles.activeLevel]}
                    onPress={() => {
                      // Store selected course data for other tabs to access
                      AsyncStorage.setItem('selectedCourse', JSON.stringify(course));
                      router.push('/player');
                    }}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="play" size={40} color="#ffffff" />
                  </TouchableOpacity>
                  {index === 0 && (
                    <View style={styles.badge}>
                      <Text style={styles.badgeText}>START HERE</Text>
                    </View>
                  )}
                </>
              )}
              {course.locked && (
                <View style={[styles.levelButton, styles.lockedLevel]}>
                  <Ionicons name="lock-closed" size={28} color="#9ca3af" />
                </View>
              )}
              <Text 
                numberOfLines={1}
                style={course.completed ? styles.levelLabel : course.locked ? styles.lockedLabel : styles.activeLevelLabel}
              >
                {course.title}
              </Text>
            </View>
            );
          })}

          {/* Loading state overlay ??shows over the static S-path */}
          {loading && (
            <View style={{ position: 'absolute', top: 0, left: 0, right: 0, height: height, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(15,23,42,0.5)' }}>
              <ActivityIndicator size="large" color="#3B82F6" />
              <Text style={{ marginTop: 16, color: '#9CA3AF', fontWeight: '600' }}>Loading study path...</Text>
            </View>
          )}
        </View>
      </ScrollView>

      {/* Empty state ??absolutely positioned over the scroll area, perfectly centered */}
      {!loading && courses.length === 0 && (
        <View
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            justifyContent: 'center',
            alignItems: 'center',
            padding: 32,
          }}
          pointerEvents="box-none"
        >
          <View
            style={{
              backgroundColor: 'rgba(15, 23, 42, 0.88)',
              borderRadius: 28,
              padding: 36,
              alignItems: 'center',
              borderWidth: 1,
              borderColor: 'rgba(59, 130, 246, 0.35)',
              shadowColor: '#3B82F6',
              shadowOffset: { width: 0, height: 0 },
              shadowOpacity: 0.25,
              shadowRadius: 24,
              elevation: 12,
              width: '100%',
            }}
          >
            <View style={styles.emptyIconContainer}>
              <Ionicons name="school-outline" size={48} color="#3B82F6" />
            </View>
            <Text
              style={{
                fontSize: 22,
                fontWeight: 'bold',
                color: '#FFFFFF',
                marginTop: 20,
                textAlign: 'center',
                letterSpacing: 0.3,
              }}
            >
              No Study Path Yet
            </Text>
            <Text
              style={{
                fontSize: 14,
                color: '#94A3B8',
                marginTop: 10,
                textAlign: 'center',
                lineHeight: 22,
              }}
            >
              Upload a PDF to generate your{`\n`}AI-powered learning journey
            </Text>
            <TouchableOpacity
              style={{
                marginTop: 28,
                backgroundColor: '#3B82F6',
                paddingHorizontal: 28,
                paddingVertical: 14,
                borderRadius: 14,
                flexDirection: 'row',
                alignItems: 'center',
                gap: 8,
                shadowColor: '#3B82F6',
                shadowOffset: { width: 0, height: 4 },
                shadowOpacity: 0.4,
                shadowRadius: 12,
                elevation: 8,
              }}
              onPress={() => router.push('/upload')}
              activeOpacity={0.85}
            >
              <Ionicons name="cloud-upload" size={20} color="#FFFFFF" />
              <Text style={{ fontSize: 15, fontWeight: '700', color: '#FFFFFF' }}>Upload PDF</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* ============ MODALS ============ */}

      {/* Day Streak Modal */}
      <Modal
        visible={streakVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setStreakVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Day Streak 🔥</Text>
              <TouchableOpacity onPress={() => setStreakVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.infoContainer}>
                <View style={styles.iconBox}>
                  <Text style={styles.largeIcon}>🔥</Text>
                </View>
                <Text style={styles.infoTitle}>What is Day Streak?</Text>
                <Text style={styles.infoText}>
                  Your Day Streak represents consecutive days of active learning. Every day you study, your streak grows stronger!
                </Text>

                <View style={styles.benefitsBox}>
                  <Text style={styles.benefitsTitle}>Benefits:</Text>
                  <Text style={styles.benefitItem}>💎 Earn bonus diamonds daily</Text>
                  <Text style={styles.benefitItem}>🔥 Build consistent study habits</Text>
                  <Text style={styles.benefitItem}>⭐ Unlock special achievements</Text>
                  <Text style={styles.benefitItem}>✨ Multiply your rewards</Text>
                </View>

                <View style={styles.tipBox}>
                  <Ionicons name="information-circle" size={20} color="#3b82f6" />
                  <Text style={styles.tipText}>Tip: Study at least once per day to keep your streak going!</Text>
                </View>

                <View style={styles.streakInfo}>
                  <Text style={styles.streakLabel}>Current Streak:</Text>
                  <Text style={styles.streakCount}>5 Days</Text>
                  <Text style={styles.streakSubText}>Keep it going! 🔥</Text>
                </View>
              </View>
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.closeModalButton}
                onPress={() => setStreakVisible(false)}
              >
                <Text style={styles.closeModalButtonText}>Got it!</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Diamonds Modal */}
      <Modal
        visible={diamondVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setDiamondVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>💎 Diamond Store</Text>
              <TouchableOpacity onPress={() => setDiamondVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.infoContainer}>
                <View style={styles.diamondDisplayBox}>
                  <Ionicons name="diamond" size={28} color="#3b82f6" />
                  <Text style={styles.diamondDisplayText}>Your Diamonds: <Text style={styles.diamondCount}>{diamonds}</Text></Text>
                </View>

                <Text style={styles.infoTitle}>Buy Badges</Text>
                <Text style={styles.infoText}>
                  Purchase exclusive badges with your diamonds to showcase your achievements!
                </Text>

                <View style={styles.badgesShopGrid}>
                  {badgeData.map((badge) => (
                    <View key={badge.id} style={styles.badgeShopCard}>
                      <View style={[
                        styles.badgeShopIconContainer,
                        badgesOwned.has(badge.id) && styles.badgeOwnedContainer
                      ]}>
                        <Text style={styles.badgeShopIcon}>{badge.name.split(' ')[0]}</Text>
                        {badgesOwned.has(badge.id) && (
                          <View style={styles.ownedBadge}>
                            <Ionicons name="checkmark" size={16} color="#FFFFFF" />
                          </View>
                        )}
                      </View>
                      <Text style={styles.badgeShopName}>{badge.name.substring(badge.name.indexOf(' ') + 1)}</Text>
                      <Text style={styles.badgeShopDesc}>{badge.description}</Text>
                      
                      {badgesOwned.has(badge.id) ? (
                        <TouchableOpacity style={styles.ownedButton} disabled>
                          <Text style={styles.ownedButtonText}>✅ Owned</Text>
                        </TouchableOpacity>
                      ) : (
                        <TouchableOpacity 
                          style={[
                            styles.buyButton,
                            diamonds < badge.price && styles.buyButtonDisabled
                          ]}
                          onPress={() => handleBuyBadge(badge.id, badge.price)}
                          disabled={diamonds < badge.price}
                        >
                          <Text style={styles.buyButtonText}>{badge.price} 💎</Text>
                        </TouchableOpacity>
                      )}
                    </View>
                  ))}
                </View>

                <View style={styles.benefitsBox}>
                  <Text style={styles.benefitsTitle}>How to Earn Diamonds:</Text>
                  <Text style={styles.benefitItem}>📖 Complete lessons (+10)</Text>
                  <Text style={styles.benefitItem}>🔥 Maintain day streaks (+50)</Text>
                  <Text style={styles.benefitItem}>📚 Finish flashcard sets (+25)</Text>
                  <Text style={styles.benefitItem}>🏆 Achieve milestones (+100)</Text>
                </View>
              </View>
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.closeModalButton}
                onPress={() => setDiamondVisible(false)}
              >
                <Text style={styles.closeModalButtonText}>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

// Styles are in ./styles/index_styles.ts
