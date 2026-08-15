import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MiniMax Music 3 · MLX",
  description: "Create complete songs with synced lyrics, privately and locally on Apple Silicon with MiniMax-Music3.",
  icons: {
    icon: "/amma-live-icon.png",
    apple: "/amma-live-icon.png",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
