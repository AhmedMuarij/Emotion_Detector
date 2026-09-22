import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EmotionAI – Real-Time Facial Expression Recognition",
  description:
    "AI-powered facial expression recognition platform. Classifies visible facial expressions in real time using a CNN trained on FER2013. Privacy-first: no images stored.",
  keywords: [
    "facial expression recognition",
    "emotion detection",
    "computer vision",
    "AI",
    "machine learning",
    "FER2013",
  ],
  robots: "index, follow",
  openGraph: {
    title: "EmotionAI – Facial Expression Recognition",
    description:
      "Real-time facial expression classification using deep learning.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
