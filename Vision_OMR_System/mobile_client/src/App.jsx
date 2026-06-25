/**
 * App.jsx
 * -------
 * Root component of the Vision OMR mobile client.
 *
 * State machine:
 *   idle ──► scanning ──► uploading ──► results ──► idle
 *                └──────────────────────────────────────┘
 *                          (re-scan)
 */

import React, { useState, useCallback } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  StyleSheet,
  StatusBar,
  Alert,
} from 'react-native';

import CameraScanner from './components/CameraScanner';
import ResultsModal  from './components/ResultsModal';
import { evaluateSheet, submitResults } from './services/api';

// ── Screens ────────────────────────────────────────────────────────────────

const SCREEN = {
  IDLE:      'idle',
  SCANNING:  'scanning',
  RESULTS:   'results',
};

// ── Component ──────────────────────────────────────────────────────────────

const App = () => {
  const [screen,  setScreen]  = useState(SCREEN.SCANNING);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleCapture = useCallback(async (photo) => {
    setLoading(true);
    setScreen(SCREEN.RESULTS); // show modal with spinner immediately

    try {
      const data = await evaluateSheet(photo);
      setResults(data);
    } catch (err) {
      Alert.alert('Evaluation Error', err.message);
      setScreen(SCREEN.SCANNING);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!results) return;
    try {
      await submitResults(results);
      Alert.alert('Submitted', 'Results saved successfully.', [
        { text: 'OK', onPress: () => { setResults(null); setScreen(SCREEN.SCANNING); } },
      ]);
    } catch (err) {
      Alert.alert('Submit Error', err.message);
    }
  }, [results]);

  const handleRescan = useCallback(() => {
    setResults(null);
    setScreen(SCREEN.SCANNING);
  }, []);

  const handleCloseModal = useCallback(() => {
    if (!loading) handleRescan();
  }, [loading, handleRescan]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0a0a14" />

      {/* App header */}
      <View style={styles.header}>
        <Text style={styles.logo}>Vision OMR</Text>
        <Text style={styles.subtitle}>Point camera at answer sheet</Text>
      </View>

      {/* Camera view */}
      <View style={styles.cameraWrapper}>
        <CameraScanner
          onCapture={handleCapture}
          onError={err => Alert.alert('Camera Error', err.message)}
        />
      </View>

      {/* Results modal */}
      <ResultsModal
        visible={screen === SCREEN.RESULTS}
        results={results}
        loading={loading}
        onSubmit={handleSubmit}
        onRescan={handleRescan}
        onClose={handleCloseModal}
      />
    </SafeAreaView>
  );
};

// ── Styles ─────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root:          { flex: 1, backgroundColor: '#0a0a14' },

  header: {
    paddingHorizontal: 20,
    paddingVertical:   14,
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e2e',
  },
  logo:          { color: '#e2e8f0', fontSize: 20, fontWeight: '800', letterSpacing: 0.5 },
  subtitle:      { color: '#64748b', fontSize: 12, marginTop: 2 },

  cameraWrapper: { flex: 1 },
});

export default App;
