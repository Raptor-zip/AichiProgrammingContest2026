/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                dark: {
                    bg: '#1a1a2e',
                    surface: '#16213e',
                    accent: '#0f3460',
                },
                primary: {
                    DEFAULT: '#533483',
                    hover: '#6b4397',
                    active: '#3d2564',
                },
                secondary: {
                    DEFAULT: '#e94560',
                    hover: '#c42847',
                }
            },
            fontFamily: {
                sans: ['"Noto Sans CJK JP"', '"Yu Gothic"', '"Meiryo"', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
