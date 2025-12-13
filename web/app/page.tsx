"use client";

import { useState } from "react";

export default function Home() {
  const [scenario, setScenario] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [response, setResponse] = useState<string | null>(null);

  const handleAnalyze = () => {
    if (!scenario.trim()) return;
    
    setIsAnalyzing(true);
    // Simulate API call - replace with actual API call later
    setTimeout(() => {
      setResponse("Based on the Indian Constitution, your scenario involves fundamental rights protected under Article 19 (freedom of speech and expression) and Article 21 (protection of life and personal liberty). The specific application depends on the circumstances and any reasonable restrictions that may apply.");
      setIsAnalyzing(false);
    }, 2000);
  };

  const sampleArticles = [
    {
      articleNumber: "Article 19",
      title: "Protection of certain rights regarding freedom of speech, etc.",
      text: "(1) All citizens shall have the right—\n(a) to freedom of speech and expression;\n(b) to assemble peaceably and without arms;\n(c) to form associations or unions or co-operative societies;\n(d) to move freely throughout the territory of India;\n(e) to reside and settle in any part of the territory of India; and\n(f) to practise any profession, or to carry on any occupation, trade or business."
    },
    {
      articleNumber: "Article 21",
      title: "Protection of life and personal liberty",
      text: "No person shall be deprived of his life or personal liberty except according to procedure established by law."
    },
    {
      articleNumber: "Article 21A",
      title: "Right to education",
      text: "The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine."
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <main className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            LawLagGaye
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Your AI-powered legal assistant that knows the Indian Constitution like the back of its digital hand. 
            Ask us anything, and we'll find the relevant articles faster than you can say "objection, your honor!"
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8 mb-8 border border-slate-200 dark:border-slate-700">
          <div className="space-y-4">
            <label htmlFor="scenario" className="block text-lg font-semibold text-slate-700 dark:text-slate-200">
              Describe Your Scenario
            </label>
            <textarea
              id="scenario"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              placeholder="e.g., My neighbor's loud music at 2 AM is driving me crazy. Can I do something about it? Or maybe: I want to start a protest march in my city. What are my rights?"
              className="w-full h-32 px-4 py-3 border-2 border-slate-300 dark:border-slate-600 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 dark:focus:ring-indigo-800 outline-none resize-none text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-700 placeholder-slate-400 dark:placeholder-slate-500"
            />
            <button
              onClick={handleAnalyze}
              disabled={!scenario.trim() || isAnalyzing}
              className="w-full py-4 px-6 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-[0.98]"
            >
              {isAnalyzing ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </span>
              ) : (
                "🔍 Analyze & Advise"
              )}
            </button>
          </div>
        </div>

        {/* Response Section */}
        {response && (
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8 mb-8 border border-slate-200 dark:border-slate-700">
            <h2 className="text-2xl font-bold mb-4 text-slate-800 dark:text-slate-100">Legal Analysis</h2>
            <div className="prose prose-slate dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap">
                {response}
              </p>
            </div>
          </div>
        )}

        {/* Sample Articles Section */}
        <div className="space-y-6">
          <h2 className="text-3xl font-bold text-center text-slate-800 dark:text-slate-100 mb-8">
            Referenced Articles
          </h2>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {sampleArticles.map((article, index) => (
              <div
                key={index}
                className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700 hover:shadow-xl transition-shadow duration-200"
              >
                <div className="mb-4">
                  <h3 className="text-xl font-bold text-indigo-600 dark:text-indigo-400 mb-2">
                    {article.articleNumber}
                  </h3>
                  <h4 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-3">
                    {article.title}
                  </h4>
                </div>
                <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed whitespace-pre-line">
                  {article.text}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Footer Note */}
        <div className="mt-12 text-center text-slate-500 dark:text-slate-400 text-sm">
          <p>
            ⚖️ This is an AI-powered tool. Always consult with a qualified legal professional for official legal advice.
          </p>
        </div>
      </main>
    </div>
  );
}
