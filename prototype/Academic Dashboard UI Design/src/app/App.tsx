import { AnalysisConfiguration } from "@/app/components/AnalysisConfiguration";
import { ResultsArea } from "@/app/components/ResultsArea";

export default function App() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="px-12 py-8">
          <div className="flex items-baseline gap-4">
            <h1 className="text-foreground tracking-tight" style={{ fontSize: '2rem', fontWeight: 600 }}>
              Household Financial Mobility System
            </h1>
            <span className="text-muted-foreground text-sm bg-secondary px-3 py-1 rounded-full">
              Prototype
            </span>
          </div>
          <p className="text-muted-foreground mt-3 max-w-3xl">
            An analytical framework for studying household economic transitions and financial well-being patterns
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="px-12 py-12">
        <div className="grid grid-cols-[380px_1fr] gap-8">
          {/* Left Sidebar - Configuration */}
          <div className="sticky top-8 self-start">
            <AnalysisConfiguration />
          </div>

          {/* Right Content - Results */}
          <div>
            <ResultsArea />
          </div>
        </div>
      </div>
    </div>
  );
}
