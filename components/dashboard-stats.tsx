"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, AlertTriangle, DollarSign, Shield, Activity } from "lucide-react"

interface DashboardStatsProps {
  analytics: {
    totalTransactions: number
    suspiciousTransactions: number
    fraudRate: number
    totalAmount: number
    suspiciousAmount: number
  }
}

export function DashboardStats({ analytics }: DashboardStatsProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount)
  }

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
      <Card className="fintech-card overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 bg-gradient-to-r from-primary/10 to-primary/5">
          <CardTitle className="text-sm font-semibold text-primary">Total Transactions</CardTitle>
          <div className="p-2 rounded-lg bg-primary/10">
            <Activity className="h-5 w-5 text-primary" />
          </div>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="text-3xl font-bold text-foreground">{analytics.totalTransactions.toLocaleString()}</div>
          <div className="flex items-center mt-2 text-sm">
            <TrendingUp className="h-4 w-4 mr-1 text-success" />
            <span className="text-success font-medium">+12%</span>
            <span className="text-muted-foreground ml-1">from last week</span>
          </div>
        </CardContent>
      </Card>

      <Card className="fintech-card overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 bg-gradient-to-r from-destructive/10 to-destructive/5">
          <CardTitle className="text-sm font-semibold text-destructive">Suspicious Transactions</CardTitle>
          <div className="p-2 rounded-lg bg-destructive/10">
            <AlertTriangle className="h-5 w-5 text-destructive" />
          </div>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="text-3xl font-bold text-destructive">{analytics.suspiciousTransactions}</div>
          <div className="flex items-center mt-2 text-sm">
            <TrendingDown className="h-4 w-4 mr-1 text-success" />
            <span className="text-success font-medium">-3%</span>
            <span className="text-muted-foreground ml-1">from last week</span>
          </div>
        </CardContent>
      </Card>

      <Card className="fintech-card overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 bg-gradient-to-r from-accent/10 to-accent/5">
          <CardTitle className="text-sm font-semibold text-accent">Fraud Rate</CardTitle>
          <div className="p-2 rounded-lg bg-accent/10">
            <Shield className="h-5 w-5 text-accent" />
          </div>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="text-3xl font-bold text-foreground">{analytics.fraudRate}%</div>
          <div className="flex items-center mt-2 text-sm">
            <span className="text-muted-foreground">Industry avg:</span>
            <span className="text-foreground font-medium ml-1">8.2%</span>
          </div>
        </CardContent>
      </Card>

      <Card className="fintech-card overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 bg-gradient-to-r from-secondary/10 to-secondary/5">
          <CardTitle className="text-sm font-semibold text-secondary">Total Amount</CardTitle>
          <div className="p-2 rounded-lg bg-secondary/10">
            <DollarSign className="h-5 w-5 text-secondary" />
          </div>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="text-3xl font-bold text-foreground">{formatCurrency(analytics.totalAmount)}</div>
          <div className="flex items-center mt-2 text-sm">
            <span className="text-muted-foreground">Suspicious:</span>
            <span className="text-destructive font-medium ml-1">{formatCurrency(analytics.suspiciousAmount)}</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
