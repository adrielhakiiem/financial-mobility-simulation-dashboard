import { Card } from "@/app/components/ui/card";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { TrendingUp, Users, DollarSign, Home } from "lucide-react";

const mobilityData = [
  { year: "2020", upward: 32, stable: 48, downward: 20 },
  { year: "2021", upward: 38, stable: 44, downward: 18 },
  { year: "2022", upward: 42, stable: 41, downward: 17 },
  { year: "2023", upward: 45, stable: 39, downward: 16 },
  { year: "2024", upward: 48, stable: 37, downward: 15 },
];

const incomeData = [
  { quartile: "Q1", median: 28500, mean: 32100 },
  { quartile: "Q2", median: 45200, mean: 47800 },
  { quartile: "Q3", median: 62300, mean: 65400 },
  { quartile: "Q4", median: 89700, mean: 105200 },
];

const summaryMetrics = [
  { 
    label: "Upward Mobility Rate", 
    value: "48%", 
    change: "+6% from 2020",
    icon: TrendingUp,
    color: "text-accent"
  },
  { 
    label: "Households Analyzed", 
    value: "12,847", 
    change: "National sample",
    icon: Home,
    color: "text-primary"
  },
  { 
    label: "Median Income Change", 
    value: "+8.2%", 
    change: "Year over year",
    icon: DollarSign,
    color: "text-chart-3"
  },
  { 
    label: "Stable Households", 
    value: "37%", 
    change: "-11% from 2020",
    icon: Users,
    color: "text-chart-2"
  },
];

export function ResultsArea() {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="mb-6 text-foreground tracking-tight">Results</h2>
        <p className="text-muted-foreground mb-8 max-w-3xl">
          This dashboard presents conceptual results for demonstration purposes. 
          The analysis framework is designed to track household financial mobility patterns 
          across multiple dimensions and time periods.
        </p>
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-4 gap-6">
        {summaryMetrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <Card key={index} className="p-6 bg-card border-border">
              <div className="flex items-start justify-between mb-4">
                <div className={`p-2.5 rounded-lg bg-secondary ${metric.color}`}>
                  <Icon className="h-5 w-5" />
                </div>
              </div>
              <div className="space-y-1">
                <p className="text-muted-foreground">
                  {metric.label}
                </p>
                <p className="text-foreground tracking-tight" style={{ fontSize: '1.75rem', fontWeight: 600 }}>
                  {metric.value}
                </p>
                <p className="text-muted-foreground" style={{ fontSize: '0.875rem' }}>
                  {metric.change}
                </p>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="p-8 bg-card border-border">
          <h3 className="mb-6 text-foreground">Financial Mobility Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mobilityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="year" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  border: '1px solid #e2e8f0',
                  borderRadius: '0.5rem'
                }} 
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="upward" 
                stroke="#5c7ea6" 
                strokeWidth={2.5}
                name="Upward Mobility (%)"
              />
              <Line 
                type="monotone" 
                dataKey="stable" 
                stroke="#8b9fb8" 
                strokeWidth={2.5}
                name="Stable (%)"
              />
              <Line 
                type="monotone" 
                dataKey="downward" 
                stroke="#3d5a80" 
                strokeWidth={2.5}
                name="Downward Mobility (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card className="p-8 bg-card border-border">
          <h3 className="mb-6 text-foreground">Income Distribution by Quartile</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={incomeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="quartile" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff', 
                  border: '1px solid #e2e8f0',
                  borderRadius: '0.5rem'
                }} 
              />
              <Legend />
              <Bar dataKey="median" fill="#5c7ea6" name="Median Income ($)" />
              <Bar dataKey="mean" fill="#8b9fb8" name="Mean Income ($)" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Summary Table */}
      <Card className="p-8 bg-card border-border">
        <h3 className="mb-6 text-foreground">Key Findings Summary</h3>
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4 pb-4 border-b border-border">
            <div className="text-muted-foreground">Category</div>
            <div className="text-muted-foreground">Observation</div>
            <div className="text-muted-foreground">Confidence</div>
          </div>
          <div className="grid grid-cols-3 gap-4 py-4 border-b border-border">
            <div className="text-foreground">Mobility Trajectory</div>
            <div className="text-foreground/80">Consistent upward trend across cohorts</div>
            <div className="text-accent">High (95%)</div>
          </div>
          <div className="grid grid-cols-3 gap-4 py-4 border-b border-border">
            <div className="text-foreground">Income Inequality</div>
            <div className="text-foreground/80">Q4 shows significant divergence from median</div>
            <div className="text-accent">High (92%)</div>
          </div>
          <div className="grid grid-cols-3 gap-4 py-4 border-b border-border">
            <div className="text-foreground">Geographical Factors</div>
            <div className="text-foreground/80">Urban-rural disparities remain significant</div>
            <div className="text-accent">Medium (78%)</div>
          </div>
          <div className="grid grid-cols-3 gap-4 py-4">
            <div className="text-foreground">Temporal Stability</div>
            <div className="text-foreground/80">5-year patterns show increasing stability</div>
            <div className="text-accent">Medium (81%)</div>
          </div>
        </div>
      </Card>

      {/* Methodology Note */}
      <Card className="p-6 bg-secondary/50 border-border">
        <div className="flex items-start gap-4">
          <div className="p-2 rounded-full bg-primary/10">
            <svg 
              className="h-5 w-5 text-primary" 
              fill="none" 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth="2" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h4 className="text-foreground mb-2">Prototype Notice</h4>
            <p className="text-muted-foreground leading-relaxed">
              This interface represents a conceptual design for a Final Year Project. 
              All data displayed is simulated for demonstration purposes and does not reflect 
              real household financial information. The system architecture is designed to 
              support multiple analytical frameworks and methodologies.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
