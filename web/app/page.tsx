"use client";

import { useState } from "react";

interface Article {
  article_number: string;
  part_number?: string;
  part_title?: string;
  score: number;
  distance: number;
}

interface ApiResponse {
  answer: string;
  articles: Article[];
  enhanced_query?: string;
  retrieval_attempts: number;
  answer_attempts: number;
  checker_flags?: any;
  evaluator_flags?: any;
}

export default function Home() {
  const [scenario, setScenario] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [response, setResponse] = useState<string | null>(null);
  const [articles, setArticles] = useState<Article[]>([]);
  const [error, setError] = useState<string | null>(null);

  const API_ENDPOINT = process.env.NEXT_PUBLIC_API_ENDPOINT || "http://localhost:8000";

  const handleAnalyze = async () => {
    if (!scenario.trim()) return;
    
    setIsAnalyzing(true);
    setError(null);
    setResponse(null);
    setArticles([]);
    
    try {
      const response = await fetch(`${API_ENDPOINT}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: scenario,
          min_score: 0.7,
          max_results: 200,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      setResponse(data.answer);
      setArticles(data.articles || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to fetch response from API";
      setError(errorMessage);
      console.error("API Error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <main className="container mx-auto px-4 py-6 max-w-6xl">
        {/* Sticky Header */}
        <header className="sticky top-0 z-20 bg-white/90 backdrop-blur border-b border-slate-200 mb-8">
          <div className="max-w-5xl mx-auto px-4 py-4 flex flex-col items-center gap-3">
            <div className="inline-flex items-center justify-center px-8 py-4 rounded-3xl maroon-texture shadow-md">
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight">
                LawLagGaye
              </h1>
            </div>
            <p className="text-base sm:text-lg md:text-xl text-slate-700 text-center max-w-2xl">
              <span className="maroon-texture-text font-semibold">
                Your playful AI-powered legal buddy
              </span>{" "}
              that knows the Indian Constitution like the back of its digital hand. Ask us anything, and we'll
              fetch the right articles faster than you can say &quot;objection, your honor!&quot;
            </p>
          </div>
        </header>

        {/* Input Section */}
        <div className="bg-white rounded-3xl card-soft-shadow p-8 mb-8 border border-slate-200">
          <div className="space-y-4">
            <label
              htmlFor="scenario"
              className="block text-lg font-semibold text-slate-800 maroon-texture-text"
            >
              Describe Your Scenario
            </label>
            <textarea
              id="scenario"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              placeholder="e.g., My neighbor's loud music at 2 AM is driving me crazy. Can I do something about it? Or maybe: I want to start a protest march in my city. What are my rights?"
              className="w-full h-32 px-4 py-3 border-2 border-slate-300 rounded-xl focus:border-[#7b1b2b] focus:ring-2 focus:ring-[#7b1b2b22] outline-none resize-none text-slate-800 bg-white placeholder-slate-400"
            />
            <div className="flex justify-center pt-2">
              <div className="relative inline-flex flex-col items-center group">
                <span className="mb-1 text-3xl text-[#7b1b2b] transform -translate-y-2 transition-transform duration-300 ease-out group-hover:translate-y-0">
                  🔨
                </span>
                <button
                  onClick={handleAnalyze}
                  disabled={!scenario.trim() || isAnalyzing}
                  className="inline-flex items-center gap-2 rounded-full px-6 py-2 text-sm font-semibold text-white bg-[#7b1b2b] shadow-md hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed group-hover:translate-y-0.5"
                >
                  {isAnalyzing ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg
                        className="animate-spin h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        ></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                      </svg>
                      Hammering out advice...
                    </span>
                  ) : (
                    <>
                      <span>Drop the hammer</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Error Section */}
        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-3xl card-soft-shadow p-8 mb-8">
            <h2 className="text-2xl font-bold mb-4 text-red-800">Error</h2>
            <p className="text-red-700 leading-relaxed">
              {error}
            </p>
            <p className="text-red-600 text-sm mt-2">
              Make sure the API server is running at {API_ENDPOINT}
            </p>
          </div>
        )}

        {/* Response Section */}
        {response && (
          <div className="bg-white rounded-3xl card-soft-shadow p-8 mb-8 border border-slate-200">
            <h2 className="text-2xl font-bold mb-4 maroon-texture-text">Legal Analysis</h2>
            <div className="prose prose-slate dark:prose-invert max-w-none">
              <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                {response}
              </p>
            </div>
          </div>
        )}

        {/* Referenced Articles Section */}
        {articles.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-center maroon-texture-text mb-8">
              Referenced Articles ({articles.length})
            </h2>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {articles.map((article, index) => (
                <div
                  key={index}
                  className="bg-white rounded-2xl card-soft-shadow p-6 border border-slate-200 hover:shadow-xl transition-shadow duration-200"
                >
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold uppercase tracking-wide maroon-texture-text">
                        Article {article.article_number}
                      </h3>
                      <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
                        {(article.score * 100).toFixed(0)}% match
                      </span>
                    </div>
                    {article.part_number && (
                      <p className="text-xs text-slate-500 mb-1">
                        Part {article.part_number}
                        {article.part_title && ` - ${article.part_title}`}
                      </p>
                    )}
                  </div>
                  <div className="text-slate-600 text-sm">
                    <p className="font-medium text-slate-800 mb-2">
                      Relevance Score: {(article.score * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500">
                      This article was referenced in the analysis above.
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer Note */}
        <div className="mt-12 text-center text-slate-500 text-sm">
          <p>
            ⚖️ This is an AI-powered tool. Always consult with a qualified legal professional for official legal advice.
          </p>
        </div>
      </main>
    </div>
  );
}
