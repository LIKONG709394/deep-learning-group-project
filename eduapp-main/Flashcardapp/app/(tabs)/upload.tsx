import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Platform,
  Modal,
  TextInput,
  Dimensions,
  Linking,
} from 'react-native';
import styles from '@/constants/tab-styles/upload_styles';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as DocumentPicker from 'expo-document-picker';
import { File, Directory, Paths } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { processPDFAndGenerateContent, processImageAndGenerateContent, processUrlAndGenerateContent } from '../../utils/gemini';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

export default function UploadScreen() {
  const router = useRouter();
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<any>(null);
  const [progress, setProgress] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Modal states
  const [scanModalVisible, setScanModalVisible] = useState(false);
  const [linkModalVisible, setLinkModalVisible] = useState(false);
  const [scannedLink, setScannedLink] = useState('');

  const handlePickDocument = async () => {
    try {
      // For web platform, use native file input
      if (Platform.OS === 'web') {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'application/pdf,.pdf';
        input.onchange = (e: any) => {
          const file = e.target.files?.[0];
          if (file) {
            if (file.size > 25 * 1024 * 1024) {
              Alert.alert('File Too Large', 'Please select a file smaller than 25MB.');
              return;
            }
            setSelectedFile({
              name: file.name,
              size: file.size,
              uri: URL.createObjectURL(file),
              mimeType: file.type,
            });
            Alert.alert('File Selected ✓', `${file.name} is ready to upload!`);
          }
        };
        input.click();
        return;
      }

      // For mobile platforms, use DocumentPicker
      const result = await DocumentPicker.getDocumentAsync({
        type: 'application/pdf',
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const file = result.assets[0];
        if (file.size && file.size > 25 * 1024 * 1024) {
          Alert.alert('File Too Large', 'Please select a file smaller than 25MB.');
          return;
        }
        setSelectedFile(file);
        Alert.alert('File Selected ✓', `${file.name} is ready to upload!`);
      }
    } catch (error) {
      console.error('Document picker error:', error);
      Alert.alert('Error', 'Failed to pick document. Please try again.');
    }
  };

  const handleScanTextbook = () => {
    setScanModalVisible(true);
  };

  // Close modal first, wait for animation to finish, then open camera
  const handleStartCamera = async () => {
    setScanModalVisible(false);
    await new Promise(resolve => setTimeout(resolve, 600));
    await launchCamera();
  };

  const launchCamera = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Camera Permission Required',
        'Please allow camera access in your device settings to scan textbooks.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Open Settings',
            onPress: () => Linking.openSettings(),
          },
        ]
      );
      return;
    }

    const pickerResult = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.85,
      base64: true,
      allowsEditing: true,
      aspect: [4, 3],
    });

    if (pickerResult.canceled || !pickerResult.assets?.[0]) return;

    const asset = pickerResult.assets[0];
    if (!asset.base64) {
      Alert.alert('Error', 'Could not read image data. Please try again.');
      return;
    }

    const apiKey = process.env.EXPO_PUBLIC_POE_API_KEY;
    if (!apiKey || apiKey === 'your_poe_api_key_here') {
      Alert.alert('API Key Required', 'Please configure your Poe API key in the .env file to use AI features.');
      return;
    }

    setUploading(true);
    setProgress('Reading text from image with AI...');

    try {
      const imageName = `Scanned Page ${new Date().toLocaleDateString()}`;
      const mimeType = asset.mimeType || 'image/jpeg';

      setProgress('Extracting text from image...');
      const scanResult = await processImageAndGenerateContent(asset.base64, mimeType, imageName);

      setProgress('Building study materials...');
      await new Promise(resolve => setTimeout(resolve, 300));

      const existingDataStr = await AsyncStorage.getItem('studyData');
      const courseNode = {
        id: Date.now(),
        title: imageName,
        completed: false,
        locked: false,
        pdfUri: asset.uri,
        pdfName: imageName + '.jpg',
        flashcards: scanResult.flashcards,
        studyPath: scanResult.studyPath,
        shortVideos: scanResult.shortVideos,
      };

      let combinedData: { courses: any[] } = { courses: [courseNode] };
      if (existingDataStr) {
        const existingData = JSON.parse(existingDataStr);
        combinedData = { courses: [...(existingData.courses || []), courseNode] };
      }

      await AsyncStorage.setItem('studyData', JSON.stringify(combinedData));
      await AsyncStorage.setItem('hasGeneratedContent', 'true');

      setUploading(false);
      setProgress('');
      Alert.alert(
        'Scan Complete! 🎉',
        `AI analyzed your scanned page and created a new course!\n\n` +
        `• ${scanResult.flashcards.length} flashcards\n` +
        `• ${scanResult.studyPath.length} study levels\n` +
        `• ${scanResult.shortVideos.length} video summaries\n\n` +
        `Check the home screen to start learning!`,
        [{ text: 'View Path', onPress: () => router.push('/') }]
      );
    } catch (error: any) {
      setUploading(false);
      setProgress('');
      Alert.alert(
        'Processing Failed',
        error.message || 'Failed to process image. Please check your API key and try again.'
      );
    }
  };

  const handlePasteLink = () => {
    setLinkModalVisible(true);
    setScannedLink('');
  };

  const handleProcessLink = async () => {
    const url = scannedLink.trim();
    if (!url) {
      Alert.alert('Error', 'Please paste a valid URL');
      return;
    }

    // Basic URL validation
    try { new URL(url); } catch {
      Alert.alert('Invalid URL', 'Please enter a full URL starting with https://');
      return;
    }

    const apiKey = process.env.EXPO_PUBLIC_POE_API_KEY;
    if (!apiKey || apiKey === 'your_poe_api_key_here') {
      Alert.alert('API Key Required', 'Please configure your Poe API key in the .env file to use AI features.');
      return;
    }

    setLinkModalVisible(false);
    setUploading(true);
    setProgress('Fetching webpage content...');

    try {
      setProgress('Extracting text from page...');
      const result = await processUrlAndGenerateContent(url);

      setProgress('Building study materials...');
      await new Promise(resolve => setTimeout(resolve, 300));

      const hostname = (() => {
        try { return new URL(url).hostname.replace(/^www\./, ''); }
        catch { return url; }
      })();

      const existingDataStr = await AsyncStorage.getItem('studyData');
      const courseNode = {
        id: Date.now(),
        title: hostname,
        completed: false,
        locked: false,
        pdfUri: url,
        pdfName: hostname,
        flashcards: result.flashcards,
        studyPath: result.studyPath,
        shortVideos: result.shortVideos,
      };

      let combinedData: { courses: any[] } = { courses: [courseNode] };
      if (existingDataStr) {
        const existingData = JSON.parse(existingDataStr);
        combinedData = { courses: [...(existingData.courses || []), courseNode] };
      }

      await AsyncStorage.setItem('studyData', JSON.stringify(combinedData));
      await AsyncStorage.setItem('hasGeneratedContent', 'true');

      setUploading(false);
      setProgress('');
      setScannedLink('');
      Alert.alert(
        'Done! 🎉',
        `AI scraped "${hostname}" and created a new course!\n\n` +
        `• ${result.flashcards.length} flashcards\n` +
        `• ${result.studyPath.length} study levels\n` +
        `• ${result.shortVideos.length} video summaries\n\n` +
        `Check the home screen to start learning!`,
        [{ text: 'View Path', onPress: () => router.push('/') }]
      );
    } catch (error: any) {
      setUploading(false);
      setProgress('');
      Alert.alert(
        'Processing Failed',
        error.message || 'Failed to process the URL. Please try a different link.'
      );
    }
  };

  const handleUpload = async () => {
    try {
      if (!selectedFile) {
        Alert.alert('No File Selected', 'Please select a PDF file first, or the feature is in demo mode.');
        return;
      }

      setUploading(true);
      setProgress('Uploading PDF...');

      // Check if API key is configured
      const apiKey = process.env.EXPO_PUBLIC_POE_API_KEY;
      if (!apiKey || apiKey === 'your_poe_api_key_here') {
        Alert.alert(
          'API Key Required',
          'Please configure your Poe API key in the .env file to use AI features.',
          [{ text: 'OK' }]
        );
        setUploading(false);
        setProgress('');
        return;
      }

      setProgress('Saving PDF...');

      // Persist the PDF file to a permanent location (mobile only)
      let permanentPdfUri: string | null = null;
      if (Platform.OS !== 'web' && selectedFile.uri) {
        try {
          const pdfsDir = new Directory(Paths.document, 'pdfs');
          if (!pdfsDir.exists) {
            pdfsDir.create({ intermediates: true });
          }
          const safeFileName = selectedFile.name.replace(/[^a-zA-Z0-9._-]/g, '_');
          const destFile = new File(pdfsDir, Date.now() + '_' + safeFileName);
          const sourceFile = new File(selectedFile.uri);
          sourceFile.copy(destFile);
          permanentPdfUri = destFile.uri;
        } catch (fsError) {
          console.warn('Could not save PDF permanently:', fsError);
          permanentPdfUri = selectedFile.uri;
        }
      } else {
        // On web, keep the blob URL for the current session
        permanentPdfUri = selectedFile.uri;
      }

      setProgress('Extracting content from PDF...');
      
      // Process PDF with Gemini AI
      const result = await processPDFAndGenerateContent(
        selectedFile.uri,
        selectedFile.name
      );

      setProgress('Generating flashcards...');
      await new Promise(resolve => setTimeout(resolve, 500));

      setProgress('Creating study path...');
      await new Promise(resolve => setTimeout(resolve, 500));

      setProgress('Generating video summaries...');
      await new Promise(resolve => setTimeout(resolve, 500));

      // Load existing data and append new content
      const existingDataStr = await AsyncStorage.getItem('studyData');
      
      // Create a single course node for this PDF
      const courseNode = {
        id: Date.now(), // Unique ID based on upload time
        title: selectedFile.name.replace('.pdf', ''),
        completed: false,
        locked: false,
        pdfUri: permanentPdfUri,
        pdfName: selectedFile.name,
        flashcards: result.flashcards,
        studyPath: result.studyPath,
        shortVideos: result.shortVideos,
      };

      let combinedData = {
        courses: [courseNode],
      };

      if (existingDataStr) {
        const existingData = JSON.parse(existingDataStr);
        combinedData = {
          courses: [...(existingData.courses || []), courseNode],
        };
      }

      // Save accumulated content to AsyncStorage
      await AsyncStorage.setItem('studyData', JSON.stringify(combinedData));
      await AsyncStorage.setItem('hasGeneratedContent', 'true');

      setUploading(false);
      setProgress('');
      setSelectedFile(null);

      // Calculate totals across all courses
      const totalFlashcards = combinedData.courses.reduce((sum: number, c: any) => sum + (c.flashcards?.length || 0), 0);
      const totalLevels = combinedData.courses.reduce((sum: number, c: any) => sum + (c.studyPath?.length || 0), 0);
      const totalVideos = combinedData.courses.reduce((sum: number, c: any) => sum + (c.shortVideos?.length || 0), 0);

      Alert.alert(
        'Success! 🎉',
        `AI has analyzed "${selectedFile.name}" and created a new course!\n\n` +
        `This course includes:\n` +
        `• ${result.flashcards.length} flashcards\n` +
        `• ${result.studyPath.length} study levels\n` +
        `• ${result.shortVideos.length} video summaries\n\n` +
        `You now have ${combinedData.courses.length} course(s) with ${totalFlashcards} total flashcards!\n\n` +
        `Check the home screen to start learning!`,
        [{ text: 'View Path', onPress: () => router.push('/') }]
      );
    } catch (error: any) {
      console.error('Upload error:', error);
      setUploading(false);
      setProgress('');
      
      Alert.alert(
        'Processing Failed',
        error.message || 'Failed to process PDF. Please check your API key and try again.',
        [{ text: 'OK' }]
      );
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>New Study Path</Text>
        <Text style={styles.subtitle}>
          Upload your lecture slides to generate a new map.
        </Text>

        {/* Upload Area */}
        <TouchableOpacity
          style={[styles.uploadBox, selectedFile && styles.uploadBoxSelected]}
          activeOpacity={0.8}
          disabled={uploading}
          onPress={handlePickDocument}
        >
          <View style={styles.uploadIcon}>
            <Ionicons 
              name={selectedFile ? "document" : "cloud-upload-outline"} 
              size={32} 
              color={selectedFile ? "#16a34a" : "#7c3aed"}
            />
          </View>
          <View style={styles.uploadTextContainer}>
            <Text style={[styles.uploadTitle, selectedFile && styles.uploadTitleSelected]}>
              {selectedFile ? selectedFile.name : 'Tap to Upload PDF'}
            </Text>
            <Text style={styles.uploadSubtitle}>
              {selectedFile 
                ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB` 
                : 'Max size: 25MB'}
            </Text>
          </View>
          {selectedFile && (
            <TouchableOpacity 
              onPress={(e) => {
                e.stopPropagation();
                setSelectedFile(null);
              }}
              style={styles.clearButton}
            >
              <Ionicons name="close-circle" size={24} color="#ef4444" />
            </TouchableOpacity>
          )}
        </TouchableOpacity>

        {/* Alternative Options */}
        <View style={styles.optionsContainer}>
          <TouchableOpacity 
            style={styles.optionButton} 
            activeOpacity={0.7}
            onPress={handleScanTextbook}
          >
            <Ionicons name="camera-outline" size={20} color="#6b7280" />
            <Text style={styles.optionText}>Scan Textbook</Text>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.optionButton} 
            activeOpacity={0.7}
            onPress={handlePasteLink}
          >
            <Ionicons name="link-outline" size={20} color="#6b7280" />
            <Text style={styles.optionText}>Paste Link</Text>
          </TouchableOpacity>
        </View>

        {/* Generate Button */}
        <TouchableOpacity
          style={[
            styles.generateButton, 
            uploading && styles.generateButtonDisabled,
            selectedFile && styles.generateButtonActive
          ]}
          onPress={handleUpload}
          disabled={uploading}
          activeOpacity={0.8}
        >
          {uploading ? (
            <>
              <ActivityIndicator color="#ffffff" size="small" />
              <Text style={styles.generateButtonText}>{progress || 'Processing...'}</Text>
            </>
          ) : (
            <>
              <Ionicons name="sparkles" size={20} color="#ffffff" />
              <Text style={styles.generateButtonText}>
                {selectedFile ? 'Generate with AI' : 'Select PDF First'} ✨
              </Text>
            </>
          )}
        </TouchableOpacity>
      </View>

      {/* ===== MODALS ===== */}

      {/* Scan Textbook Modal */}
      <Modal
        visible={scanModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setScanModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Scan Textbook</Text>
              <TouchableOpacity onPress={() => setScanModalVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <View style={styles.modalBody}>
              <View style={styles.scanInfoContainer}>
                <Ionicons name="camera" size={48} color="#3B82F6" />
                <Text style={styles.scanInfoTitle}>Open Camera</Text>
                <Text style={styles.scanInfoDescription}>
                  Point your camera at a textbook or document page to scan and extract content. Our AI will automatically recognize and process the text.
                </Text>

                <View style={styles.featuresList}>
                  <View style={styles.featureItem}>
                    <Ionicons name="checkmark-circle" size={20} color="#3B82F6" />
                    <Text style={styles.featureText}>Instant text recognition</Text>
                  </View>
                  <View style={styles.featureItem}>
                    <Ionicons name="checkmark-circle" size={20} color="#3B82F6" />
                    <Text style={styles.featureText}>Auto image enhancement</Text>
                  </View>
                  <View style={styles.featureItem}>
                    <Ionicons name="checkmark-circle" size={20} color="#3B82F6" />
                    <Text style={styles.featureText}>Supports multiple languages</Text>
                  </View>
                </View>
              </View>
            </View>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.cancelModalButton}
                onPress={() => setScanModalVisible(false)}
              >
                <Text style={styles.cancelModalButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={styles.startCameraButton}
                onPress={handleStartCamera}
              >
                <Ionicons name="camera" size={20} color="#FFFFFF" />
                <Text style={styles.startCameraButtonText}>Start Camera</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Paste Link Modal */}
      <Modal
        visible={linkModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setLinkModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Paste Link</Text>
              <TouchableOpacity onPress={() => setLinkModalVisible(false)}>
                <Ionicons name="close" size={24} color="#FFFFFF" />
              </TouchableOpacity>
            </View>

            <View style={styles.modalBody}>
              <Text style={styles.inputLabel}>Website URL</Text>
              <TextInput
                style={styles.urlInput}
                placeholder="https://example.com"
                placeholderTextColor="#6B7280"
                value={scannedLink}
                onChangeText={setScannedLink}
                keyboardType="url"
                autoCapitalize="none"
              />

              <Text style={styles.linkDescription}>
                📖 Paste the URL of any website to extract and process its content. Our AI will scrape the webpage and generate study materials from it.
              </Text>

              <View style={styles.supportedSitesContainer}>
                <Text style={styles.supportedSitesTitle}>Supported Sources:</Text>
                <View style={styles.supportedSite}>
                  <Ionicons name="checkmark" size={16} color="#3B82F6" />
                  <Text style={styles.supportedSiteText}>Articles & Blog Posts</Text>
                </View>
                <View style={styles.supportedSite}>
                  <Ionicons name="checkmark" size={16} color="#3B82F6" />
                  <Text style={styles.supportedSiteText}>Wikipedia Pages</Text>
                </View>
                <View style={styles.supportedSite}>
                  <Ionicons name="checkmark" size={16} color="#3B82F6" />
                  <Text style={styles.supportedSiteText}>Educational Platforms</Text>
                </View>
              </View>
            </View>

            <View style={styles.modalFooter}>
              <TouchableOpacity 
                style={styles.cancelModalButton}
                onPress={() => setLinkModalVisible(false)}
              >
                <Text style={styles.cancelModalButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.processLinkButton, !scannedLink.trim() && styles.processLinkButtonDisabled]}
                onPress={handleProcessLink}
                disabled={!scannedLink.trim()}
              >
                <Ionicons name="link" size={20} color="#FFFFFF" />
                <Text style={styles.processLinkButtonText}>Process Link</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

// Styles are in ./styles/upload_styles.ts
