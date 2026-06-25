/**
 * CameraScanner.jsx
 * -----------------
 * React Native camera module for capturing OMR sheet photos.
 *
 * Features:
 *  - Real-time camera preview via react-native-vision-camera
 *  - JPEG compression with configurable quality (default 80%)
 *    to keep payload sizes manageable without sacrificing readability
 *  - Resolution safeguard: downscales frames exceeding MAX_DIMENSION
 *  - Flash toggle and focus-lock on tap
 *  - Emits { uri, width, height, size } to parent via onCapture callback
 */

import React, { useRef, useState, useCallback } from 'react';
import {
  View,
  TouchableOpacity,
  StyleSheet,
  Text,
  Alert,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import {
  Camera,
  useCameraDevices,
  useCameraPermission,
} from 'react-native-vision-camera';
import ImageResizer from '@bam.tech/react-native-image-resizer';

// ── Constants ────────────────────────────────────────────────────────────
const JPEG_QUALITY     = 80;   // 0-100: quality vs file-size balance
const MAX_DIMENSION    = 1920; // px – downscale above this to prevent OOM
const ASPECT_RATIO     = 4 / 3;

// ── Component ────────────────────────────────────────────────────────────

/**
 * @param {function} onCapture   – Called with { uri, width, height, size }
 * @param {function} onError     – Called with an Error object
 */
const CameraScanner = ({ onCapture, onError }) => {
  const camera          = useRef(null);
  const devices         = useCameraDevices();
  const device          = devices.back;
  const { hasPermission, requestPermission } = useCameraPermission();

  const [flash, setFlash]         = useState('off');
  const [capturing, setCapturing] = useState(false);

  // ── Permission gate ────────────────────────────────────────────────────
  const ensurePermission = useCallback(async () => {
    if (hasPermission) return true;
    const granted = await requestPermission();
    if (!granted) {
      Alert.alert(
        'Camera Permission Required',
        'Please grant camera access in Settings to scan OMR sheets.',
      );
    }
    return granted;
  }, [hasPermission, requestPermission]);

  // ── Capture & compress ─────────────────────────────────────────────────
  const handleCapture = useCallback(async () => {
    const permitted = await ensurePermission();
    if (!permitted || !camera.current) return;

    try {
      setCapturing(true);

      // 1. Take photo
      const photo = await camera.current.takePhoto({
        flash,
        qualityPrioritization: 'quality',
        skipMetadata: true,
      });

      // 2. Resolution safeguard – downscale if too large
      const srcW = photo.width;
      const srcH = photo.height;
      let targetW = srcW;
      let targetH = srcH;

      if (Math.max(srcW, srcH) > MAX_DIMENSION) {
        const scale  = MAX_DIMENSION / Math.max(srcW, srcH);
        targetW = Math.round(srcW * scale);
        targetH = Math.round(srcH * scale);
      }

      // 3. Compress to JPEG
      const resized = await ImageResizer.createResizedImage(
        `file://${photo.path}`,
        targetW,
        targetH,
        'JPEG',
        JPEG_QUALITY,
        0,           // rotation (EXIF-corrected by lib)
        undefined,   // output path (temp)
        false,       // keep metadata
        { onlyScaleDown: true, mode: 'contain' },
      );

      onCapture?.({
        uri:    resized.uri,
        width:  resized.width,
        height: resized.height,
        size:   resized.size,
      });
    } catch (err) {
      onError?.(err);
      Alert.alert('Capture Failed', err.message);
    } finally {
      setCapturing(false);
    }
  }, [camera, flash, ensurePermission, onCapture, onError]);

  // ── Render ─────────────────────────────────────────────────────────────
  if (!device) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6C63FF" />
        <Text style={styles.waitText}>Initialising camera…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera preview */}
      <Camera
        ref={camera}
        style={styles.camera}
        device={device}
        isActive
        photo
        enableZoomGesture
      />

      {/* Overlay guides */}
      <View style={styles.overlay} pointerEvents="none">
        <View style={styles.scanFrame} />
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        {/* Flash toggle */}
        <TouchableOpacity
          style={styles.iconBtn}
          onPress={() => setFlash(f => (f === 'off' ? 'on' : 'off'))}
          accessibilityLabel="Toggle flash"
        >
          <Text style={styles.iconText}>{flash === 'on' ? '⚡' : '🔦'}</Text>
        </TouchableOpacity>

        {/* Shutter */}
        <TouchableOpacity
          style={[styles.shutter, capturing && styles.shutterDisabled]}
          onPress={handleCapture}
          disabled={capturing}
          accessibilityLabel="Capture OMR sheet"
        >
          {capturing
            ? <ActivityIndicator color="#fff" />
            : <View style={styles.shutterInner} />
          }
        </TouchableOpacity>

        {/* Spacer */}
        <View style={styles.iconBtn} />
      </View>
    </View>
  );
};

// ── Styles ────────────────────────────────────────────────────────────────
const { width: SCREEN_W } = Dimensions.get('window');

const styles = StyleSheet.create({
  container:       { flex: 1, backgroundColor: '#000' },
  camera:          { flex: 1 },
  center:          { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#0a0a0a' },
  waitText:        { color: '#aaa', marginTop: 12, fontSize: 14 },

  overlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scanFrame: {
    width:        SCREEN_W * 0.85,
    height:       (SCREEN_W * 0.85) * ASPECT_RATIO,
    borderWidth:  2,
    borderColor:  'rgba(108,99,255,0.8)',
    borderRadius: 8,
  },

  controls: {
    flexDirection:   'row',
    alignItems:      'center',
    justifyContent:  'space-between',
    paddingHorizontal: 36,
    paddingBottom:   40,
    paddingTop:      20,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  iconBtn:          { width: 48, alignItems: 'center' },
  iconText:         { fontSize: 26 },

  shutter: {
    width:           72,
    height:          72,
    borderRadius:    36,
    backgroundColor: '#6C63FF',
    alignItems:      'center',
    justifyContent:  'center',
    shadowColor:     '#6C63FF',
    shadowOpacity:   0.6,
    shadowRadius:    12,
    elevation:       8,
  },
  shutterDisabled:  { opacity: 0.5 },
  shutterInner: {
    width:           52,
    height:          52,
    borderRadius:    26,
    backgroundColor: '#fff',
  },
});

export default CameraScanner;
