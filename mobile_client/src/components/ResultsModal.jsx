/**
 * ResultsModal.jsx
 * ----------------
 * Full-screen verification overlay that displays the OMR grading results
 * returned by the backend API.
 *
 * Layout:
 *  ┌─────────────────────────────┐
 *  │  Summary banner             │
 *  │  (filled / empty / review)  │
 *  ├─────────────────────────────┤
 *  │  Scrollable bubble list     │
 *  │  • Each row shows bbox,     │
 *  │    state chip, fill ratio   │
 *  ├─────────────────────────────┤
 *  │  [ Re-scan ]  [ Submit ]    │
 *  └─────────────────────────────┘
 */

import React, { useCallback } from 'react';
import {
  Modal,
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ActivityIndicator,
} from 'react-native';
import { C, RADIUS } from '../tokens';

// ── Chip colours per state — mapped to design tokens ─────────────────────
const STATE_COLORS = {
  filled:    { bg: C.greenBg,  border: C.green,  text: C.green  },
  empty:     { bg: C.bg,       border: C.textMute, text: C.textMute },
  ambiguous: { bg: C.amberBg,  border: C.amber,  text: C.amber  },
};

// ── Sub-components ────────────────────────────────────────────────────────

const SummaryBanner = ({ data }) => (
  <View style={styles.banner}>
    <Text style={styles.bannerTitle}>Grading Summary</Text>
    <View style={styles.bannerRow}>
      <Stat label="Filled"    value={data.filled_count}    color={C.green}    />
      <Stat label="Empty"     value={data.empty_count}     color={C.textMute} />
      <Stat label="Ambiguous" value={data.ambiguous_count} color={C.amber}    />
    </View>
    {data.needs_manual_review && (
      <View style={styles.reviewAlert}>
        <Text style={styles.reviewAlertText}>
          Manual review required for {data.ambiguous_count} bubble
          {data.ambiguous_count !== 1 ? 's' : ''}.
        </Text>
      </View>
    )}
    <Text style={styles.processingTime}>
      Processed in {data.processing_time_ms} ms
    </Text>
  </View>
);

const Stat = ({ label, value, color }) => (
  <View style={styles.stat}>
    <Text style={[styles.statValue, { color }]}>{value}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

const BubbleRow = ({ item, index }) => {
  const chip = STATE_COLORS[item.state] ?? STATE_COLORS.ambiguous;
  return (
    <View style={styles.row}>
      <Text style={styles.rowIndex}>#{index + 1}</Text>
      <View style={styles.rowBody}>
        <View style={[styles.chip, { backgroundColor: chip.bg, borderColor: chip.border }]}>
          <Text style={[styles.chipText, { color: chip.text }]}>
            {item.state.toUpperCase()}
          </Text>
        </View>
        <Text style={styles.rowDetail}>
          Fill: {(item.fill_ratio * 100).toFixed(1)}%
          {item.needs_review ? '  Review' : ''}
        </Text>
      </View>
      <Text style={styles.rowConf}>{(item.confidence * 100).toFixed(0)}%</Text>
    </View>
  );
};

// ── Main component ────────────────────────────────────────────────────────

/**
 * @param {boolean}  visible     – Modal visibility flag
 * @param {object}   results     – API response from POST /evaluate
 * @param {boolean}  loading     – Show spinner while request is in-flight
 * @param {function} onSubmit    – Called when user confirms results
 * @param {function} onRescan    – Called when user requests a new scan
 * @param {function} onClose     – Called to close modal without action
 */
const ResultsModal = ({ visible, results, loading, onSubmit, onRescan, onClose }) => {
  const keyExtractor = useCallback((_, i) => String(i), []);
  const renderItem   = useCallback(
    ({ item, index }) => <BubbleRow item={item} index={index} />,
    [],
  );

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <SafeAreaView style={styles.container}>
        {/* ── Navy header ──────────────────────────────────────────────── */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>OMR Results</Text>
          <TouchableOpacity onPress={onClose} accessibilityLabel="Close">
            <Text style={styles.closeBtn}>✕</Text>
          </TouchableOpacity>
        </View>

        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={C.navy} />
            <Text style={styles.loadingText}>Analysing sheet…</Text>
          </View>
        ) : results ? (
          <>
            <SummaryBanner data={results} />
            <FlatList
              data={results.bubbles}
              keyExtractor={keyExtractor}
              renderItem={renderItem}
              contentContainerStyle={styles.list}
              showsVerticalScrollIndicator={false}
            />
          </>
        ) : (
          <View style={styles.loadingContainer}>
            <Text style={styles.loadingText}>No results yet.</Text>
          </View>
        )}

        {/* ── Action bar ───────────────────────────────────────────────── */}
        {!loading && results && (
          <View style={styles.actions}>
            {/* Outlined secondary: Re-scan / Flag */}
            <TouchableOpacity
              style={[styles.btn, styles.btnOutline]}
              onPress={onRescan}
              accessibilityLabel="Re-scan sheet"
            >
              <Text style={styles.btnOutlineText}>Re-scan</Text>
            </TouchableOpacity>
            {/* Solid navy primary: Submit */}
            <TouchableOpacity
              style={[styles.btn, styles.btnPrimary]}
              onPress={onSubmit}
              accessibilityLabel="Submit results"
            >
              <Text style={styles.btnPrimaryText}>Submit</Text>
            </TouchableOpacity>
          </View>
        )}
      </SafeAreaView>
    </Modal>
  );
};

// ── Styles ────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: C.bg },

  // ── Navy header ──────────────────────────────────────────────────────────
  header: {
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'center',
    paddingHorizontal: 20,
    paddingVertical:   16,
    backgroundColor:   C.navy,
  },
  headerTitle: { color: C.textOnNavy, fontSize: 18, fontWeight: '700' },
  closeBtn:    { color: C.navySub, fontSize: 20, padding: 4 },

  // ── Summary banner card ──────────────────────────────────────────────────
  banner: {
    margin:           16,
    padding:          16,
    backgroundColor:  C.surface,
    borderRadius:     RADIUS.card,
    borderWidth:      1,
    borderColor:      C.border,
  },
  bannerTitle:  { color: C.textHead, fontSize: 15, fontWeight: '600', marginBottom: 12 },
  bannerRow:    { flexDirection: 'row', justifyContent: 'space-around', marginBottom: 8 },
  stat:         { alignItems: 'center' },
  statValue:    { fontSize: 26, fontWeight: '800' },
  statLabel:    { color: C.textMute, fontSize: 12, marginTop: 2 },

  // Review alert
  reviewAlert: {
    marginTop:       10,
    padding:         10,
    backgroundColor: C.amberBg,
    borderRadius:    8,
    borderLeftWidth: 3,
    borderLeftColor: C.amber,
  },
  reviewAlertText: { color: C.amber, fontSize: 13 },
  processingTime:  { color: C.textMute, fontSize: 11, marginTop: 8, textAlign: 'right' },

  // ── Bubble list ──────────────────────────────────────────────────────────
  list: { paddingHorizontal: 16, paddingBottom: 16 },
  row: {
    flexDirection:     'row',
    alignItems:        'center',
    paddingVertical:   10,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  rowIndex:  { color: C.textMute, width: 32, fontSize: 12 },
  rowBody:   { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 10 },
  chip: {
    paddingHorizontal: 8,
    paddingVertical:   3,
    borderRadius:      4,
    borderWidth:       1,
  },
  chipText:   { fontSize: 11, fontWeight: '700' },
  rowDetail:  { color: C.textMute, fontSize: 12 },
  rowConf:    { color: C.textMute, fontSize: 12, width: 36, textAlign: 'right' },

  // ── Loading ──────────────────────────────────────────────────────────────
  loadingContainer: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  loadingText:      { color: C.textMute, marginTop: 16, fontSize: 14 },

  // ── Action bar ───────────────────────────────────────────────────────────
  actions: {
    flexDirection:   'row',
    gap:             12,
    padding:         20,
    borderTopWidth:  1,
    borderTopColor:  C.border,
    backgroundColor: C.surface,
  },
  btn:             { flex: 1, paddingVertical: 14, borderRadius: RADIUS.btn, alignItems: 'center' },
  btnPrimary:      { backgroundColor: C.navy },
  btnPrimaryText:  { color: C.textOnNavy, fontWeight: '700', fontSize: 15 },
  btnOutline:      { borderWidth: 1, borderColor: C.navy },
  btnOutlineText:  { color: C.navy, fontWeight: '600', fontSize: 15 },
});

export default ResultsModal;
