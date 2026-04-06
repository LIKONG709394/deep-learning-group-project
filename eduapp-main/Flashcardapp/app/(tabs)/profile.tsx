import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Modal,
  TextInput,
  Switch,
  Dimensions,
  Image,
  Alert,
} from 'react-native';
import styles from '@/constants/tab-styles/profile_styles';
import { Ionicons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as ImagePicker from 'expo-image-picker';

const { width } = Dimensions.get('window');

export default function ProfileScreen() {
  const router = useRouter();
  const [userName, setUserName] = useState('Student Name');
  const [userEmail, setUserEmail] = useState('student@example.com');
  
  // Profile image
  const [profileImage, setProfileImage] = useState<string | null>(null);

  // Modal states
  const [editProfileVisible, setEditProfileVisible] = useState(false);
  const [notificationsVisible, setNotificationsVisible] = useState(false);
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [helpVisible, setHelpVisible] = useState(false);
  const [aboutVisible, setAboutVisible] = useState(false);
  const [streakVisible, setStreakVisible] = useState(false);
  const [diamondVisible, setDiamondVisible] = useState(false);
  const [badgesVisible, setBadgesVisible] = useState(false);

  // Form states
  const [editName, setEditName] = useState(userName);
  const [editEmail, setEditEmail] = useState(userEmail);
  
  // Settings states
  const [notifEnabled, setNotifEnabled] = useState(true);
  const [emailReminders, setEmailReminders] = useState(true);
  const [pushNotif, setPushNotif] = useState(true);
  const [darkMode, setDarkMode] = useState(true);

  // Diamond and Badge states
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
    loadBadgeData();
    AsyncStorage.getItem('profileImage').then((uri) => {
      if (uri) setProfileImage(uri);
    });
  }, []);

  const handleEditProfile = () => {
    setEditName(userName);
    setEditEmail(userEmail);
    setEditProfileVisible(true);
  };

  const handleSaveProfile = () => {
    setUserName(editName);
    setUserEmail(editEmail);
    setEditProfileVisible(false);
  };

  const handleNotifications = () => {
    setNotificationsVisible(true);
  };

  const handleSettings = () => {
    setSettingsVisible(true);
  };

  const handleHelp = () => {
    setHelpVisible(true);
  };

  const handleAbout = () => {
    setAboutVisible(true);
  };

  const handlePickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission Required', 'Please allow access to your photo library to set a profile picture.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled && result.assets.length > 0) {
      const uri = result.assets[0].uri;
      setProfileImage(uri);
      await AsyncStorage.setItem('profileImage', uri);
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.avatarContainer} onPress={handlePickImage} activeOpacity={0.8}>
            {profileImage ? (
              <Image source={{ uri: profileImage }} style={styles.avatarImage} />
            ) : (
              <Text style={styles.avatar}>{"👨‍🎓"}</Text>
            )}
            <View style={styles.avatarEditBadge}>
              <Ionicons name="camera" size={14} color="#FFFFFF" />
            </View>
          </TouchableOpacity>
          <Text style={styles.name}>{userName}</Text>
          <Text style={styles.email}>{userEmail}</Text>
        </View>

        {/* Stats */}
        <View style={styles.statsContainer}>
          <TouchableOpacity 
            style={styles.statCard}
            activeOpacity={0.7}
            onPress={() => setStreakVisible(true)}
          >
            <Ionicons name="flame" size={32} color="#f97316" />
            <Text style={styles.statValue}>5</Text>
            <Text style={styles.statLabel}>Day Streak</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={styles.statCard}
            activeOpacity={0.7}
            onPress={() => setDiamondVisible(true)}
          >
            <Ionicons name="diamond" size={32} color="#3b82f6" />
            <Text style={styles.statValue}>{diamonds}</Text>
            <Text style={styles.statLabel}>Diamonds</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={styles.statCard}
            activeOpacity={0.7}
            onPress={() => setBadgesVisible(true)}
          >
            <Ionicons name="trophy" size={32} color="#fbbf24" />
            <Text style={styles.statValue}>{badgesPurchased}</Text>
            <Text style={styles.statLabel}>Badges Earned</Text>
          </TouchableOpacity>
        </View>

        {/* Menu Items */}
        <View style={styles.menuSection}>
          <TouchableOpacity 
            style={styles.menuItem} 
            activeOpacity={0.7}
            onPress={handleEditProfile}
          >
            <View style={styles.menuIconContainer}>
              <Ionicons name="person-outline" size={24} color="#7c3aed" />
            </View>
            <Text style={styles.menuText}>Edit Profile</Text>
            <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.menuItem} 
            activeOpacity={0.7}
            onPress={handleNotifications}
          >
            <View style={styles.menuIconContainer}>
              <Ionicons name="notifications-outline" size={24} color="#7c3aed" />
            </View>
            <Text style={styles.menuText}>Notifications</Text>
            <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.menuItem} 
            activeOpacity={0.7}
            onPress={handleSettings}
          >
            <View style={styles.menuIconContainer}>
              <Ionicons name="settings-outline" size={24} color="#7c3aed" />
            </View>
            <Text style={styles.menuText}>Settings</Text>
            <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.menuItem} 
            activeOpacity={0.7}
            onPress={handleHelp}
          >
            <View style={styles.menuIconContainer}>
              <Ionicons name="help-circle-outline" size={24} color="#7c3aed" />
            </View>
            <Text style={styles.menuText}>Help & Support</Text>
            <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.menuItem} 
            activeOpacity={0.7}
            onPress={handleAbout}
          >
            <View style={styles.menuIconContainer}>
              <Ionicons name="information-circle-outline" size={24} color="#7c3aed" />
            </View>
            <Text style={styles.menuText}>About</Text>
            <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
          </TouchableOpacity>
        </View>


      </ScrollView>

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

      {/* Badges Earned Modal */}
      <Modal
        visible={badgesVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setBadgesVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>My Badges 🏆</Text>
              <TouchableOpacity onPress={() => setBadgesVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.infoContainer}>
                <View style={styles.iconBox}>
                  <Text style={styles.largeIcon}>🏆</Text>
                </View>
                <Text style={styles.infoTitle}>Your Achievements</Text>
                <Text style={styles.infoText}>
                  These are the badges you've collected and earned!
                </Text>

                <View style={styles.badgesShopGrid}>
                  {badgeData.map((badge) => (
                    badgesOwned.has(badge.id) && (
                      <View key={badge.id} style={styles.badgeShopCard}>
                        <View style={styles.badgeOwnedContainer}>
                          <Text style={styles.badgeShopIcon}>{badge.name.split(' ')[0]}</Text>
                          <View style={styles.ownedBadge}>
                            <Ionicons name="checkmark" size={16} color="#FFFFFF" />
                          </View>
                        </View>
                        <Text style={styles.badgeShopName}>{badge.name.substring(badge.name.indexOf(' ') + 1)}</Text>
                        <Text style={styles.badgeShopDesc}>{badge.description}</Text>
                        <View style={styles.ownedButton}>
                          <Text style={styles.ownedButtonText}>✅ Owned</Text>
                        </View>
                      </View>
                    )
                  ))}
                </View>

                {badgesPurchased === 0 && (
                  <View style={styles.emptyStateBox}>
                    <Text style={styles.emptyStateText}>You haven't purchased any badges yet!</Text>
                    <Text style={styles.emptyStateSubText}>Go to the Diamonds section to buy badges.</Text>
                  </View>
                )}

                <View style={styles.statsBox}>
                  <Text style={styles.statLabelSmall}>Your Progress:</Text>
                  <View style={styles.progressRow}>
                    <Text style={styles.progressLabel}>Badges Owned:</Text>
                    <Text style={styles.progressValue}>{badgesPurchased} / {badgeData.length}</Text>
                  </View>
                  <View style={styles.progressBar}>
                    <View style={[styles.progressFill, { width: `${(badgesPurchased / badgeData.length) * 100}%` }]} />
                  </View>
                  <Text style={styles.progressText}>
                    {badgesPurchased === badgeData.length 
                      ? '🎉 You own all badges!' 
                      : `Collect ${badgeData.length - badgesPurchased} more badges!`}
                  </Text>
                </View>
              </View>
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.closeModalButton}
                onPress={() => setBadgesVisible(false)}
              >
                <Text style={styles.closeModalButtonText}>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
      <Modal
        visible={editProfileVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setEditProfileVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Edit Profile</Text>
              <TouchableOpacity onPress={() => setEditProfileVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <Text style={styles.inputLabel}>Name</Text>
              <TextInput
                style={styles.textInput}
                placeholder="Enter your name"
                placeholderTextColor="#6B7280"
                value={editName}
                onChangeText={setEditName}
              />

              <Text style={styles.inputLabel}>Email</Text>
              <TextInput
                style={styles.textInput}
                placeholder="Enter your email"
                placeholderTextColor="#6B7280"
                value={editEmail}
                onChangeText={setEditEmail}
                keyboardType="email-address"
              />
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.cancelButton}
                onPress={() => setEditProfileVisible(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={styles.saveButton}
                onPress={handleSaveProfile}
              >
                <Text style={styles.saveButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Notifications Modal */}
      <Modal
        visible={notificationsVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setNotificationsVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Notifications</Text>
              <TouchableOpacity onPress={() => setNotificationsVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.settingRow}>
                <View>
                  <Text style={styles.settingLabel}>Enable Notifications</Text>
                  <Text style={styles.settingDescription}>Receive study reminders</Text>
                </View>
                <Switch
                  value={notifEnabled}
                  onValueChange={setNotifEnabled}
                  trackColor={{ false: '#3f3f46', true: '#3b82f6' }}
                  thumbColor={notifEnabled ? '#60a5fa' : '#9ca3af'}
                />
              </View>

              <View style={styles.settingRow}>
                <View>
                  <Text style={styles.settingLabel}>Email Reminders</Text>
                  <Text style={styles.settingDescription}>Get progress updates via email</Text>
                </View>
                <Switch
                  value={emailReminders}
                  onValueChange={setEmailReminders}
                  trackColor={{ false: '#3f3f46', true: '#3b82f6' }}
                  thumbColor={emailReminders ? '#60a5fa' : '#9ca3af'}
                />
              </View>

              <View style={styles.settingRow}>
                <View>
                  <Text style={styles.settingLabel}>Push Notifications</Text>
                  <Text style={styles.settingDescription}>Instant notifications on your device</Text>
                </View>
                <Switch
                  value={pushNotif}
                  onValueChange={setPushNotif}
                  trackColor={{ false: '#3f3f46', true: '#3b82f6' }}
                  thumbColor={pushNotif ? '#60a5fa' : '#9ca3af'}
                />
              </View>
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.closeModalButton}
                onPress={() => setNotificationsVisible(false)}
              >
                <Text style={styles.closeModalButtonText}>Done</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Settings Modal */}
      <Modal
        visible={settingsVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setSettingsVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Settings</Text>
              <TouchableOpacity onPress={() => setSettingsVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.settingRow}>
                <View>
                  <Text style={styles.settingLabel}>Dark Mode</Text>
                  <Text style={styles.settingDescription}>Use dark theme</Text>
                </View>
                <Switch
                  value={darkMode}
                  onValueChange={setDarkMode}
                  trackColor={{ false: '#3f3f46', true: '#3b82f6' }}
                  thumbColor={darkMode ? '#60a5fa' : '#9ca3af'}
                />
              </View>

              <TouchableOpacity style={styles.settingItem}>
                <View>
                  <Text style={styles.settingLabel}>Language</Text>
                  <Text style={styles.settingDescription}>English (US)</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
              </TouchableOpacity>

              <TouchableOpacity style={styles.settingItem}>
                <View>
                  <Text style={styles.settingLabel}>Study Reminder</Text>
                  <Text style={styles.settingDescription}>Daily at 9:00 AM</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
              </TouchableOpacity>

              <TouchableOpacity style={styles.settingItem}>
                <View>
                  <Text style={styles.settingLabel}>Data & Privacy</Text>
                  <Text style={styles.settingDescription}>Manage your data</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#9ca3af" />
              </TouchableOpacity>

              <TouchableOpacity 
                style={[styles.settingItem, styles.dangerItem]}
                onPress={async () => {
                  try {
                    await AsyncStorage.removeItem('studyData');
                    setSettingsVisible(false);
                    alert('All courses and study data cleared!');
                  } catch (error) {
                    alert('Error clearing data');
                  }
                }}
              >
                <View>
                  <Text style={[styles.settingLabel, styles.dangerText]}>Clear All Data</Text>
                  <Text style={styles.settingDescription}>Delete all uploaded PDFs and courses</Text>
                </View>
                <Ionicons name="trash" size={20} color="#EF4444" />
              </TouchableOpacity>
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.closeModalButton}
                onPress={() => setSettingsVisible(false)}
              >
                <Text style={styles.closeModalButtonText}>Done</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Help Modal */}
      <Modal
        visible={helpVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setHelpVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Help & Support</Text>
              <TouchableOpacity onPress={() => setHelpVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.helpSection}>
                <Text style={styles.helpSectionTitle}>Frequently Asked Questions</Text>
                <TouchableOpacity style={styles.helpItem}>
                  <Text style={styles.helpItemTitle}>How do I upload a PDF?</Text>
                  <Ionicons name="chevron-forward" size={18} color="#6B7280" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.helpItem}>
                  <Text style={styles.helpItemTitle}>How do flashcards work?</Text>
                  <Ionicons name="chevron-forward" size={18} color="#6B7280" />
                </TouchableOpacity>
                <TouchableOpacity style={styles.helpItem}>
                  <Text style={styles.helpItemTitle}>Can I export my notes?</Text>
                  <Ionicons name="chevron-forward" size={18} color="#6B7280" />
                </TouchableOpacity>
              </View>

              <View style={styles.helpSection}>
                <Text style={styles.helpSectionTitle}>Contact Us</Text>
                <Text style={styles.helpText}>{"\uD83D\uDCE7"} support@snapstudy.com</Text>
                <Text style={styles.helpText}>{"\uD83D\uDCAC"} Live Chat Available 9AM - 6PM</Text>
                <Text style={styles.helpText}>{"\uD83D\uDCDE"} +852 9765-3655</Text>
              </View>
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.closeModalButton}
                onPress={() => setHelpVisible(false)}
              >
                <Text style={styles.closeModalButtonText}>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* About Modal */}
      <Modal
        visible={aboutVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setAboutVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>About SnapStudy</Text>
              <TouchableOpacity onPress={() => setAboutVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.aboutContainer}>
                <Text style={styles.appIcon}>{"\uD83C\uDF93"}</Text>
                <Text style={styles.appName}>SnapStudy</Text>
                <Text style={styles.appVersion}>Version 1.0.0</Text>
                
                <Text style={styles.aboutDescription}>
                  An AI-powered learning platform designed to help students study smarter, not harder.
                </Text>

                <View style={styles.aboutSection}>
                  <Text style={styles.sectionTitle}>Features</Text>
                  <Text style={styles.sectionText}>{"\uD83E\uDD16"} AI-powered content generation</Text>
                  <Text style={styles.sectionText}>{"\uD83D\uDCDA"} Smart flashcards</Text>
                  <Text style={styles.sectionText}>{"\uD83D\uDDFA\uFE0F"} Structured learning paths</Text>
                  <Text style={styles.sectionText}>{"\uD83C\uDFA5"} Short video summaries</Text>
                  <Text style={styles.sectionText}>{"\uD83D\uDCCA"} Progress tracking</Text>
                </View>

                <View style={styles.aboutSection}>
                  <Text style={styles.sectionTitle}>Credits</Text>
                  <Text style={styles.sectionText}>Powered by Poe AI</Text>
                  <Text style={styles.sectionText}>Built with React Native & Expo</Text>
                </View>

                <Text style={styles.copyright}>© 2026 SnapStudy. All rights reserved.</Text>
              </View>
            </ScrollView>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.closeModalButton}
                onPress={() => setAboutVisible(false)}
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

// Styles are in ./styles/profile_styles.ts
