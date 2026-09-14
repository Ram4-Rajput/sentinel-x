/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        // Command-center palette: deep space + spectral intelligence hues.
        void: "#05070d",
        panel: "#0a0f1c",
        panel2: "#0e1526",
        grid: "#16203a",
        edge: "#1d2b4d",
        // semantic channels — kept distinct per spec (risk / uncertainty / novelty)
        risk: "#ff4d5e",
        risklow: "#3ddc97",
        uncertainty: "#a970ff",
        novelty: "#ffb020",
        observed: "#38e1ff",
        forecast: "#6b7bd6",
        ink: "#c7d2e6",
        inkdim: "#65748f",
      },
      boxShadow: {
        glow: "0 0 24px -4px rgba(56,225,255,0.4)",
        riskglow: "0 0 28px -2px rgba(255,77,94,0.45)",
      },
      keyframes: {
        gridpan: {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "40px 40px" },
        },
      },
      animation: {
        gridpan: "gridpan 8s linear infinite",
      },
    },
  },
  plugins: [],
};
