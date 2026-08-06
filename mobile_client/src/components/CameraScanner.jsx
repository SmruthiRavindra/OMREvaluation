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
import { C, RADIUS } from '../tokens';

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
        <ActivityIndicator size="large" color={C.navy} />
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
        {/* Status chip */}
        <View style={styles.statusChip}>
          <Text style={styles.statusChipText}>Detecting sheet</Text>
        </View>

        {/* Corner-bracket alignment guide */}
        <View style={styles.frameContainer}>
          {/* Top-left */}
          <View style={[styles.corner, styles.cornerTL]} />
          {/* Top-right */}
          <View style={[styles.corner, styles.cornerTR]} />
          {/* Bottom-left */}
          <View style={[styles.corner, styles.cornerBL]} />
          {/* Bottom-right */}
          <View style={[styles.corner, styles.cornerBR]} />
        </View>

        {/* Info row */}
        <View style={styles.infoRow}>
          <Text style={styles.infoText}>Align the full OMR sheet within the frame</Text>
        </View>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        {/* Flash toggle — outlined icon button */}
        <TouchableOpacity
          style={styles.iconBtn}
          onPress={() => setFlash(f => (f === 'off' ? 'on' : 'off'))}
          accessibilityLabel="Toggle flash"
        >
          <Text style={styles.iconText}>{flash === 'on' ? '⚡' : '🔦'}</Text>
        </TouchableOpacity>

        {/* Shutter — plain white circle, no glow */}
        <TouchableOpacity
          style={[styles.shutter, capturing && styles.shutterDisabled]}
          onPress={handleCapture}
          disabled={capturing}
          accessibilityLabel="Capture OMR sheet"
        >
          {capturing
            ? <ActivityIndicator color={C.textOnNavy} />
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
const FRAME_W = SCREEN_W * 0.85;
const FRAME_H = FRAME_W * ASPECT_RATIO;
const CORNER_SIZE = 24;
const CORNER_THICKNESS = 3;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera:    { flex: 1 },
  center:    { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: C.bg },
  waitText:  { color: C.textMute, marginTop: 12, fontSize: 14 },

  // ── Overlay ──────────────────────────────────────────────────────────────
  overlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },

  // Status chip above the frame
  statusChip: {
    backgroundColor: C.border,
    borderRadius: RADIUS.chip,
    paddingHorizontal: 14,
    paddingVertical: 6,
    marginBottom: 16,
  },
  statusChipText: {
    color: C.textBody,
    fontSize: 12,
    fontWeight: '500',
  },

  // Corner-bracket frame container
  frameContainer: {
    width: FRAME_W,
    height: FRAME_H,
    position: 'relative',
  },

  // Shared corner piece style
  corner: {
    position: 'absolute',
    width: CORNER_SIZE,
    height: CORNER_SIZE,
    borderColor: C.green,
  },
  // Top-left — border top + left
  cornerTL: {
    top: 0,
    left: 0,
    borderTopWidth: CORNER_THICKNESS,
    borderLeftWidth: CORNER_THICKNESS,
  },
  // Top-right — border top + right
  cornerTR: {
    top: 0,
    right: 0,
    borderTopWidth: CORNER_THICKNESS,
    borderRightWidth: CORNER_THICKNESS,
  },
  // Bottom-left — border bottom + left
  cornerBL: {
    bottom: 0,
    left: 0,
    borderBottomWidth: CORNER_THICKNESS,
    borderLeftWidth: CORNER_THICKNESS,
  },
  // Bottom-right — border bottom + right
  cornerBR: {
    bottom: 0,
    right: 0,
    borderBottomWidth: CORNER_THICKNESS,
    borderRightWidth: CORNER_THICKNESS,
  },

  // Info row below the frame
  infoRow: {
    marginTop: 16,
    paddingHorizontal: 24,
    paddingVertical: 8,
    backgroundColor: 'rgba(0,0,0,0.45)',
    borderRadius: RADIUS.btn,
  },
  infoText: {
    color: C.border,
    fontSize: 12,
    textAlign: 'center',
  },

  // ── Controls bar ─────────────────────────────────────────────────────────
  controls: {
    flexDirection:    'row',
    alignItems:       'center',
    justifyContent:   'space-between',
    paddingHorizontal: 36,
    paddingBottom:    40,
    paddingTop:       20,
    backgroundColor:  'rgba(22, 50, 79, 0.9)', // navy at 90% opacity
  },

  // Outlined icon button (flash)
  iconBtn: {
    width: 48,
    height: 48,
    borderRadius: RADIUS.chip,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconText: { fontSize: 22 },

  // Plain circular shutter — white border, white inner — no shadow/glow
  shutter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#FFFFFF',
    backgroundColor: 'transparent',
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterDisabled: { opacity: 0.5 },
  shutterInner: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#FFFFFF',
  },
});

export default CameraScanner;
