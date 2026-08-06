/**
 * tokens.js
 * ---------
 * Single source of truth for the Vision OMR design system.
 * Import from here; never hardcode colours in component files.
 */

export const C = {
  // ── Core palette ─────────────────────────────────────────────────────────
  navy:    '#16324F',   // headers, primary buttons, active nav
  bg:      '#F4F6F9',   // screen background
  surface: '#FFFFFF',   // card fills
  border:  '#E1E5EB',   // hairline borders

  // ── State colours (correctness / status only — never decorative) ─────────
  green:   '#1F7A5C',
  red:     '#C0392B',
  amber:   '#B8860B',

  // ── Semantic surface tints (1:8 opacity backgrounds for chips) ───────────
  greenBg: '#EAF5F1',
  redBg:   '#FDECEA',
  amberBg: '#FEF9E7',

  // ── Text ─────────────────────────────────────────────────────────────────
  textHead: '#16324F',   // headings — same as navy
  textBody: '#2D3748',   // body
  textMute: '#7A8799',   // secondary / muted labels
  textOnNavy: '#FFFFFF', // text that sits on navy backgrounds
  navySub: '#B8D0E8',    // subtext on navy backgrounds
};

// ── Shared shape constants ──────────────────────────────────────────────────
export const RADIUS = {
  card:   12,
  btn:    10,
  chip:   999, // full-round
};
