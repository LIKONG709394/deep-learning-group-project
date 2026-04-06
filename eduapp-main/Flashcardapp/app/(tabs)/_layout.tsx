import { Tabs } from 'expo-router';
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

import { HapticTab } from '@/components/haptic-tab';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: '#3B82F6',
        tabBarInactiveTintColor: '#6B7280',
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarStyle: {
          position: 'absolute',
          bottom: 24,
          left: 24,
          right: 24,
          height: 80,
          backgroundColor: '#232630',
          borderRadius: 30,
          borderTopWidth: 0,
          paddingBottom: 20,
          paddingTop: 15,
          shadowColor: '#000',
          shadowOffset: { width: 0, height: 10 },
          shadowOpacity: 0.5,
          shadowRadius: 30,
          elevation: 20,
          borderWidth: 1,
          borderColor: 'rgba(255, 255, 255, 0.05)',
        },
        tabBarShowLabel: false,
        tabBarIconStyle: {
          marginTop: 0,
        },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Path',
          tabBarIcon: ({ color, focused }) => (
            <View style={[styles.iconContainer, focused && styles.iconContainerActive]}>
              <Ionicons name="map" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="cards"
        options={{
          title: 'Cards',
          tabBarIcon: ({ color, focused }) => (
            <View style={[styles.iconContainer, focused && styles.iconContainerActive]}>
              <Ionicons name="layers" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="upload"
        options={{
          title: '',
          tabBarIcon: ({ color, focused }) => (
            <View style={styles.uploadButton}>
              <Ionicons name="add-circle" size={32} color={focused ? '#3B82F6' : '#6B7280'} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="shorts"
        options={{
          title: 'Shorts',
          tabBarIcon: ({ color, focused }) => (
            <View style={[styles.iconContainer, focused && styles.iconContainerActive]}>
              <Ionicons name="videocam" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="player"
        options={{
          title: 'Player',
          tabBarIcon: ({ color, focused }) => (
            <View style={[styles.iconContainer, focused && styles.iconContainerActive]}>
              <Ionicons name="musical-notes" size={24} color={color} />
            </View>
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'Profile',
          tabBarIcon: ({ color, focused }) => (
            <View style={[styles.iconContainer, focused && styles.iconContainerActive]}>
              <Ionicons name="person" size={24} color={color} />
            </View>
          ),
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  iconContainerActive: {
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 15,
    elevation: 8,
  },
  activeIndicator: {
    position: 'absolute',
    bottom: -4,
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#3B82F6',
  },
  uploadButton: {
    width: 48,
    height: 48,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
