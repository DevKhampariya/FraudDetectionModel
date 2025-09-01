"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import { ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

interface FraudChartsProps {
  analytics: {
    dailyStats: Array<{
      date: string
      total: number
      suspicious: number
      fraudRate: number
    }>
    riskDistribution: Array<{
      risk: string
      count: number
      percentage: number
    }>
    transactionTypes: Array<{
      type: string
      total: number
      suspicious: number
    }>
  }
  transactions?: Array<{
    transactionId: string
    customerId: string
    type: string
    amount: number
    riskScore: number
    timestamp: string
    status: string
    reason: string
  }>
}

const COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e"]

const CustomPieTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background border rounded-lg p-3 shadow-lg">
        <p className="font-medium">{`Risk Level: ${data.risk || "N/A"}`}</p>
        <p>{`Count: ${data.count || 0}`}</p>
        <p>{`Percentage: ${data.percentage || 0}%`}</p>
      </div>
    )
  }
  return null
}

export function FraudCharts({ analytics, transactions = [] }: FraudChartsProps) {
  return (
    <div className="grid gap-6 md:grid-cols-1">
      <Card className="shadow-sm border-l-4 border-l-orange-500">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            {"🎯 Risk Distribution"}
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <ChartContainer
            config={{
              count: {
                label: "Count",
                color: "#ef4444",
              },
            }}
            className="h-[350px] w-full"
          >
            <ResponsiveContainer width="100%" height="100%">
              <PieChart margin={{ top: 25, right: 25, bottom: 25, left: 25 }}>
                <Pie
                  data={analytics.riskDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ risk, percentage }) => `${risk}: ${percentage}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="count"
                  stroke="#fff"
                  strokeWidth={2}
                >
                  {analytics.riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <ChartTooltip content={<CustomPieTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  )
}
