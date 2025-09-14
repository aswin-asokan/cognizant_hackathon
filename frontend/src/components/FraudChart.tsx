import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface Report {
  id: number;
  filename: string;
  date: string;
  totalRecords: number;
  fraudulentRecords: number;
  anomalyType: string;
  status: string;
}

interface FraudChartProps {
  data: Report[];
}

const FraudChart: React.FC<FraudChartProps> = ({ data }) => {
  // Process data for the chart
  const anomalyCounts: Record<string, number> = {};
  
  data.forEach(report => {
    if (!anomalyCounts[report.anomalyType]) {
      anomalyCounts[report.anomalyType] = 0;
    }
    anomalyCounts[report.anomalyType] += report.fraudulentRecords;
  });

  const chartData = Object.entries(anomalyCounts).map(([name, value]) => ({
    name,
    value
  }));

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip 
            formatter={(value) => [`${value} cases`, 'Fraudulent']}
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#F9FAFB'
            }}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FraudChart;