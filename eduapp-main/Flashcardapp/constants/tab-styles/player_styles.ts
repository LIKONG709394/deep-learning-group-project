import { StyleSheet, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },

  // ── Space background (mirrors home page) ──────────────────────────────
  backgroundContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    overflow: 'hidden',
    zIndex: 0,
  },
  spaceGradient: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    backgroundColor: '#0F172A',
  },
  sunContainer: {
    position: 'absolute',
    top: -60,
    right: -60,
    width: 140,
    height: 140,
  },
  sun: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: '#FCD34D',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sunGlow: {
    position: 'absolute',
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: 'rgba(252, 211, 77, 0.3)',
    top: -10,
    left: -10,
  },
  earthContainer: {
    position: 'absolute',
    top: '35%',
    left: '-12%',
    width: 140,
    height: 140,
  },
  earth: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: '#06B6D4',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  earthDetail: {
    width: '100%',
    height: '100%',
    borderRadius: 70,
    backgroundColor: 'rgba(34, 197, 94, 0.4)',
  },
  starsContainer: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  star: {
    position: 'absolute',
    width: 3,
    height: 3,
    backgroundColor: '#FFFFFF',
    borderRadius: 2,
    shadowColor: '#FFFFFF',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 2,
    elevation: 2,
  },
  particlesContainer: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  particle: {
    position: 'absolute',
    width: 2,
    height: 2,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    borderRadius: 1,
  },

  // ── Header ──────────────────────────────────────────────────────────────
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 12,
    zIndex: 1,
  },
  headerText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#9CA3AF',
    letterSpacing: 2,
  },

  // ── CD ──────────────────────────────────────────────────────────────────
  albumContainer: {
    alignItems: 'center',
    paddingVertical: 28,
    zIndex: 1,
  },
  albumArt: {
    width: 240,
    height: 240,
    borderRadius: 120,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#7C3AED',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 40,
    elevation: 25,
  },

  // ── Course info ──────────────────────────────────────────────────────────
  infoContainer: {
    alignItems: 'center',
    paddingHorizontal: 16,
    marginBottom: 14,
    zIndex: 1,
  },
  courseTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
    textAlign: 'center',
  },
  contentCount: {
    fontSize: 13,
    color: '#9CA3AF',
  },

  // ── Progress ──────────────────────────────────────────────────────────────
  progressContainer: {
    paddingHorizontal: 24,
    marginBottom: 20,
    zIndex: 1,
    height: 4,
    backgroundColor: '#1F2128',
    marginHorizontal: 24,
    borderRadius: 2,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#3B82F6',
    borderRadius: 2,
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 6,
  },

  // ── Controls ──────────────────────────────────────────────────────────────
  controlsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 20,
    zIndex: 1,
  },
  playButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#3B82F6',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 24,
    elevation: 15,
  },

  // ── Lyrics / content area ─────────────────────────────────────────────────
  contentHeaderContainer: {
    paddingHorizontal: 24,
    marginBottom: 10,
    zIndex: 1,
  },
  contentHeader: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#6B7280',
    letterSpacing: 2,
  },
  lyricsContainer: {
    flex: 1,
    paddingHorizontal: 24,
    zIndex: 1,
  },
  lyricsContent: {
    paddingBottom: 120,
  },
  contentCard: {
    backgroundColor: 'rgba(42, 45, 58, 0.85)',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.15)',
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  contentTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#3B82F6',
    marginBottom: 12,
    textAlign: 'center',
  },
  contentText: {
    fontSize: 14,
    lineHeight: 22,
    color: '#D1D5DB',
    textAlign: 'justify',
  },
  loadingText: {
    color: '#FFFFFF',
    fontSize: 16,
    textAlign: 'center',
  },

  // ── Empty state ────────────────────────────────────────────────────────────
  emptyStateContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
    zIndex: 1,
  },
  emptyAlbumArt: {
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: '#1F2128',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 32,
    borderWidth: 2,
    borderColor: 'rgba(59,130,246,0.3)',
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 20,
    elevation: 10,
  },
  emptyTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 12,
    textAlign: 'center',
  },
  emptyDescription: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 20,
  },
  uploadButton: {
    flexDirection: 'row',
    backgroundColor: '#3B82F6',
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    gap: 8,
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 8,
  },
  uploadButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default styles;
