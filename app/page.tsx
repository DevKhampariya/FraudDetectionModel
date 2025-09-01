"use client"

import { useState, useEffect } from "react"
import { DashboardStats } from "@/components/dashboard-stats"
import { TransactionsTable } from "@/components/transactions-table"
import { FraudCharts } from "@/components/fraud-charts"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Shield,
  Brain,
  AlertTriangle,
  Home,
  CreditCard,
  Bell,
  FileText,
  Settings,
  Search,
  Menu,
  X,
  Download,
  RefreshCw,
  Activity,
  BarChart3,
  Clock,
  TrendingUp,
  Users,
  DollarSign,
} from "lucide-react"

interface Transaction {
  transactionId: string
  customerId: string
  type: string
  amount: number
  riskScore: number
  timestamp: string
  status: string
  reason: string
  location: string
  merchant: string
  category: string
  paymentMethod: string
  deviceInfo: string
  ipAddress: string
  country: string
  city: string
  step?: number
  nameOrig?: string
  oldbalanceOrg?: number
  newbalanceOrig?: number
  nameDest?: string
  oldbalanceDest?: number
  newbalanceDest?: number
  hour?: number
  day_of_week?: string
  is_weekend?: boolean
  is_night?: boolean
  avg_amount?: number
  transaction_count?: number
  amount_std?: number
  amount_to_balance_ratio?: number
  amount_zscore?: number
  is_round_amount?: boolean
  balance_drained?: boolean
  dest_transaction_count?: number
  is_new_dest?: boolean
  hourly_txn_count?: number
  balance_change_ratio?: number
  unique_destinations?: number
  unique_senders_to_dest?: number
  deviation_flag?: boolean
  high_velocity_flag?: boolean
  suspicious_time_flag?: boolean
  txn_frequency_per_day?: number
  amount_risk_score?: number
  behavioral_risk_score?: number
  isFraud?: boolean
}

interface Analytics {
  totalTransactions: number
  suspiciousTransactions: number
  fraudRate: number
  totalAmount: number
  suspiciousAmount: number
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

export default function FraudDetectionDashboard() {
  const [activeView, setActiveView] = useState("overview")
  const [analytics, setAnalytics] = useState<any>(null)
  const [transactions, setTransactions] = useState<any[]>([])
  const [currentFilter, setCurrentFilter] = useState("all")
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)
  const [refreshCount, setRefreshCount] = useState(0)
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  const [loading, setLoading] = useState(true)
  const [isTraining, setIsTraining] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const [showNotifications, setShowNotifications] = useState(false)
  const [notifications] = useState([
    {
      id: 1,
      type: "alert",
      title: "High Risk Transaction",
      message: "Transaction TXN-789456123 flagged for review",
      time: "2 minutes ago",
      read: false,
    },
    {
      id: 2,
      type: "info",
      title: "System Update",
      message: "Fraud detection model updated successfully",
      time: "1 hour ago",
      read: false,
    },
    {
      id: 3,
      type: "warning",
      title: "Unusual Pattern Detected",
      message: "Multiple transactions from same IP address",
      time: "3 hours ago",
      read: true,
    },
  ])

  useEffect(() => {
    fetchAnalytics()
    fetchTransactions(currentFilter)
  }, [])

  useEffect(() => {
    fetchTransactions(currentFilter)
  }, [currentFilter])

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null

    if (activeView === "transactions" && isRealTimeEnabled) {
      interval = setInterval(async () => {
        try {
          const newRefreshCount = refreshCount + 1
          const response = await fetch(
            `/api/transactions?filter=${currentFilter}&limit=50&refresh=${newRefreshCount}`,
            { cache: "no-store" },
          )
          if (response.ok) {
            const data = await response.json()
            setTransactions(data.transactions)
            setRefreshCount(newRefreshCount)
            setLastUpdate(new Date())
            console.log("Real-time refresh completed with refresh count:", newRefreshCount)
          }
        } catch (error) {
          console.error("Auto-refresh error:", error)
        }
      }, 30000) // 30 seconds
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [activeView, isRealTimeEnabled, currentFilter, refreshCount])

  const fetchAnalytics = async () => {
    try {
      const response = await fetch("/api/analytics", { cache: "no-store" })
      const data = await response.json()
      setAnalytics(data)
    } catch (error) {
      console.error("Error fetching analytics:", error)
    }
  }

  const fetchTransactions = async (filter: string) => {
    try {
      setLoading(true)
      const response = await fetch(`/api/transactions?filter=${filter}&limit=50`, { cache: "no-store" })
      const data = await response.json()
      setTransactions(data.transactions)
    } catch (error) {
      console.error("Error fetching transactions:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleFilterChange = (filter: string) => {
    setCurrentFilter(filter)
  }

  const handleTrainModel = async () => {
    setIsTraining(true)
    try {
      const response = await fetch("/api/train-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })

      const result = await response.json()

      if (response.ok && result.success) {
        alert(
          `Model training completed successfully!\nAccuracy: ${result.metrics.accuracy}%\nPrecision: ${result.metrics.precision}%`,
        )
        await fetchAnalytics()
      } else {
        alert(`Model training failed: ${result.message}`)
      }
    } catch (error) {
      console.error("Error training model:", error)
      alert("Error starting model training. Please try again.")
    } finally {
      setIsTraining(false)
    }
  }

  const downloadCSV = async () => {
    if (isExporting) return

    setIsExporting(true)
    try {
      const response = await fetch(`/api/transactions?filter=all&limit=999999`, { cache: "no-store" })
      const data = await response.json()

      if (!data.transactions || data.transactions.length === 0) {
        alert("No transactions available to export")
        return
      }

      const allTransactions = data.transactions

      const csvContent = [
        [
          "Transaction ID",
          "Customer ID",
          "Merchant",
          "Location",
          "Category",
          "Type",
          "Amount",
          "Risk Score",
          "Status",
          "Payment Method",
          "Device",
          "IP Address",
          "Country",
          "City",
          "Step",
          "Name Orig",
          "Old Balance Orig",
          "New Balance Orig",
          "Name Dest",
          "Old Balance Dest",
          "New Balance Dest",
          "Hour",
          "Day of Week",
          "Is Weekend",
          "Is Night",
          "Avg Amount",
          "Transaction Count",
          "Amount Std",
          "Amount to Balance Ratio",
          "Amount Z-Score",
          "Is Round Amount",
          "Balance Drained",
          "Dest Transaction Count",
          "Is New Dest",
          "Hourly Txn Count",
          "Balance Change Ratio",
          "Unique Destinations",
          "Unique Senders to Dest",
          "Deviation Flag",
          "High Velocity Flag",
          "Suspicious Time Flag",
          "Txn Frequency Per Day",
          "Amount Risk Score",
          "Behavioral Risk Score",
          "Is Fraud",
        ],
        ...allTransactions.map((t: any) => [
          t.transactionId,
          t.customerId,
          t.merchant,
          t.location,
          t.category,
          t.type,
          `$${t.amount.toLocaleString()}`,
          t.riskScore,
          t.status,
          t.paymentMethod,
          t.deviceInfo,
          t.ipAddress,
          t.country,
          t.city,
          t.step || "",
          t.nameOrig || "",
          t.oldbalanceOrg || "",
          t.newbalanceOrig || "",
          t.nameDest || "",
          t.oldbalanceDest || "",
          t.newbalanceDest || "",
          t.hour || "",
          t.day_of_week || "",
          t.is_weekend || "",
          t.is_night || "",
          t.avg_amount || "",
          t.transaction_count || "",
          t.amount_std || "",
          t.amount_to_balance_ratio || "",
          t.amount_zscore || "",
          t.is_round_amount || "",
          t.balance_drained || "",
          t.dest_transaction_count || "",
          t.is_new_dest || "",
          t.hourly_txn_count || "",
          t.balance_change_ratio || "",
          t.unique_destinations || "",
          t.unique_senders_to_dest || "",
          t.deviation_flag || "",
          t.high_velocity_flag || "",
          t.suspicious_time_flag || "",
          t.txn_frequency_per_day || "",
          t.amount_risk_score || "",
          t.behavioral_risk_score || "",
          t.isFraud || "",
        ]),
      ]
        .map((row) => row.map((cell) => `"${cell}"`).join(","))
        .join("\n")

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `fraud-transactions-full-dataset-${new Date().toISOString().split("T")[0]}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)

      alert(`Successfully exported ${allTransactions.length} transactions to CSV`)
    } catch (error) {
      console.error("Error exporting data:", error)
      alert("Error exporting data. Please try again.")
    } finally {
      setIsExporting(false)
    }
  }

  const handleRefreshData = async () => {
    if (isRefreshing) return

    const now = new Date()
    if (lastRefresh && now.getTime() - lastRefresh.getTime() < 3000) {
      alert("Please wait 3 seconds between refreshes")
      return
    }

    setIsRefreshing(true)
    const newRefreshCount = refreshCount + 1

    try {
      if (activeView === "transactions") {
        const response = await fetch(`/api/transactions?filter=${currentFilter}&limit=50&refresh=${newRefreshCount}`, {
          cache: "no-store",
        })
        if (response.ok) {
          const data = await response.json()
          setTransactions(data.transactions)
          setLastRefresh(new Date())
          setRefreshCount(newRefreshCount)
          setLastUpdate(new Date())
          console.log("Transactions refreshed successfully with refresh count:", newRefreshCount)
        } else {
          throw new Error("Failed to fetch data")
        }
      } else {
        // For other views, refresh analytics data
        const [analyticsResponse, transactionsResponse] = await Promise.all([
          fetch("/api/analytics", { cache: "no-store" }),
          fetch(`/api/transactions?filter=${currentFilter}&limit=50&refresh=${newRefreshCount}`, { cache: "no-store" }),
        ])

        if (analyticsResponse.ok && transactionsResponse.ok) {
          const analyticsData = await analyticsResponse.json()
          const transactionsData = await transactionsResponse.json()

          setAnalytics(analyticsData)
          setTransactions(transactionsData.transactions)
          setLastRefresh(new Date())
          setRefreshCount(newRefreshCount)
          console.log("All data refreshed successfully with refresh count:", newRefreshCount)
        } else {
          throw new Error("Failed to fetch data")
        }
      }
    } catch (error) {
      console.error("Error refreshing data:", error)
      alert("Error refreshing data. Please try again.")
    } finally {
      setIsRefreshing(false)
    }
  }

  const handleToggleRealTime = () => {
    const newState = !isRealTimeEnabled
    setIsRealTimeEnabled(newState)

    if ((window as any).realtimeInterval) {
      clearInterval((window as any).realtimeInterval)
      delete (window as any).realtimeInterval
    }

    if (newState) {
      const interval = setInterval(async () => {
        if (!isRefreshing) {
          await handleRefreshData()
        }
      }, 30000)
      ;(window as any).realtimeInterval = interval
    }
  }

  useEffect(() => {
    return () => {
      if ((window as any).realtimeInterval) {
        clearInterval((window as any).realtimeInterval)
      }
    }
  }, [])

  const renderContent = () => {
    switch (activeView) {
      case "overview":
        return (
          <div className="space-y-8">
            <section>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">Fraud Detection Overview</h2>
                  <p className="text-muted-foreground">Real-time monitoring and analysis of suspicious activities</p>
                </div>
              </div>
              <DashboardStats analytics={analytics} />
            </section>

            <section>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">Fraud Analytics</h2>
                  <p className="text-muted-foreground">Advanced data visualization and insights</p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadCSV}
                    disabled={isExporting}
                    className="fintech-button bg-transparent"
                  >
                    <Download className={`h-4 w-4 mr-2 ${isExporting ? "animate-spin" : ""}`} />
                    {isExporting ? "Exporting..." : "Export Full Dataset"}
                  </Button>
                </div>
              </div>
              <FraudCharts analytics={analytics} transactions={transactions} />
            </section>

            <section>
              <Card className="fintech-card">
                <CardHeader className="bg-gradient-to-r from-primary/5 to-secondary/5">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg fintech-gradient">
                        <Brain className="h-5 w-5 text-white" />
                      </div>
                      <span className="text-primary">AI Model Performance</span>
                    </CardTitle>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleTrainModel}
                      disabled={isTraining}
                      className="fintech-button bg-transparent"
                    >
                      <Brain className={`h-4 w-4 mr-2 ${isTraining ? "animate-spin" : ""}`} />
                      {isTraining ? "Training..." : "Retrain Model"}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="grid gap-6 md:grid-cols-3">
                    <div className="p-4 rounded-xl bg-primary/5 border border-primary/20">
                      <h4 className="font-semibold text-sm text-muted-foreground mb-1">Model Type</h4>
                      <p className="text-lg font-bold text-primary">Random Forest Classifier</p>
                    </div>
                    <div className="p-4 rounded-xl bg-success/5 border border-success/20">
                      <h4 className="font-semibold text-sm text-muted-foreground mb-1">Accuracy</h4>
                      <p className="text-lg font-bold text-success">94.2%</p>
                    </div>
                    <div className="p-4 rounded-xl bg-secondary/5 border border-secondary/20">
                      <h4 className="font-semibold text-sm text-muted-foreground mb-1">Last Trained</h4>
                      <p className="text-lg font-bold text-secondary">2 hours ago</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </section>

            <section>
              <div className="mb-6">
                <h2 className="text-2xl font-bold text-foreground mb-2">Quick Actions</h2>
                <p className="text-muted-foreground">Manage your fraud detection system</p>
              </div>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card
                  className="fintech-card p-4 hover:scale-105 transition-transform cursor-pointer"
                  role="button"
                  tabIndex={0}
                  aria-disabled={isTraining}
                  onClick={() => {
                    if (!isTraining) handleTrainModel()
                  }}
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" || e.key === " ") && !isTraining) {
                      e.preventDefault()
                      handleTrainModel()
                    }
                  }}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Brain className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-primary">Train Model</h3>
                      <p className="text-sm text-muted-foreground">Update AI with latest data</p>
                    </div>
                  </div>
                  <Button
                    className="w-full fintech-button bg-primary hover:bg-primary/90"
                    onClick={handleTrainModel}
                    disabled={isTraining}
                    aria-disabled={isTraining}
                  >
                    {isTraining ? "Training..." : "Start Training"}
                  </Button>
                </Card>

                <Card
                  className="fintech-card p-4 hover:scale-105 transition-transform cursor-pointer"
                  role="button"
                  tabIndex={0}
                  aria-pressed={isRealTimeEnabled}
                  onClick={handleToggleRealTime}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault()
                      handleToggleRealTime()
                    }
                  }}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2 rounded-lg bg-secondary/10">
                      <RefreshCw className="h-6 w-6 text-secondary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-secondary">Auto Refresh</h3>
                      <p className="text-sm text-muted-foreground">Enable real-time monitoring</p>
                    </div>
                  </div>
                  <Button
                    className={`w-full fintech-button ${isRealTimeEnabled ? "bg-green-600 hover:bg-green-700" : "bg-secondary hover:bg-secondary/90"}`}
                    onClick={handleToggleRealTime}
                    aria-pressed={isRealTimeEnabled}
                  >
                    {isRealTimeEnabled ? "Disable Real-time" : "Enable Real-time"}
                  </Button>
                </Card>

                <Card
                  className="fintech-card p-4 hover:scale-105 transition-transform cursor-pointer"
                  role="button"
                  tabIndex={0}
                  aria-disabled={isExporting}
                  onClick={() => {
                    if (!isExporting) downloadCSV()
                  }}
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" || e.key === " ") && !isExporting) {
                      e.preventDefault()
                      downloadCSV()
                    }
                  }}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2 rounded-lg bg-accent/10">
                      <Download className="h-6 w-6 text-accent" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-accent">Export Report</h3>
                      <p className="text-sm text-muted-foreground">Download full dataset</p>
                    </div>
                  </div>
                  <Button
                    className="w-full fintech-button bg-accent hover:bg-accent/90"
                    onClick={downloadCSV}
                    disabled={isExporting}
                    aria-disabled={isExporting}
                  >
                    {isExporting ? "Exporting..." : "Export Full CSV"}
                  </Button>
                </Card>

                <Card
                  className="fintech-card p-4 hover:scale-105 transition-transform cursor-pointer"
                  role="button"
                  tabIndex={0}
                  aria-pressed={isRealTimeEnabled}
                  onClick={handleToggleRealTime}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault()
                      handleToggleRealTime()
                    }
                  }}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="p-2 rounded-lg bg-success/10">
                      <Activity className="h-6 w-6 text-success" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-success">Real-time Monitor</h3>
                      <p className="text-sm text-muted-foreground">Toggle live monitoring</p>
                    </div>
                  </div>
                  <Button
                    className={`w-full fintech-button ${isRealTimeEnabled ? "bg-success hover:bg-success/90" : "bg-muted hover:bg-muted/90 text-muted-foreground"}`}
                    onClick={handleToggleRealTime}
                    aria-pressed={isRealTimeEnabled}
                  >
                    {isRealTimeEnabled ? "Stop Monitoring" : "Start Monitoring"}
                  </Button>
                </Card>
              </div>
            </section>
          </div>
        )

      case "transactions":
        return (
          <div className="space-y-8">
            <section>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">Transaction Monitor</h2>
                  <p className="text-muted-foreground">
                    Live transaction analysis and risk assessment
                    {isRealTimeEnabled && (
                      <span className="ml-2 inline-flex items-center gap-1 text-green-600">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        Live
                      </span>
                    )}
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant={isRealTimeEnabled ? "default" : "outline"}
                    size="sm"
                    onClick={() => setIsRealTimeEnabled(!isRealTimeEnabled)}
                    className="fintech-button"
                  >
                    <Activity className="h-4 w-4 mr-2" />
                    {isRealTimeEnabled ? "Live On" : "Live Off"}
                  </Button>
                  <Button
                    variant={currentFilter === "suspicious" ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleFilterChange("suspicious")}
                    className="fintech-button"
                  >
                    <AlertTriangle className="h-4 w-4 mr-2" />
                    Suspicious Only
                  </Button>
                  <Button
                    variant={currentFilter === "all" ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleFilterChange("all")}
                    className="fintech-button"
                  >
                    <Activity className="h-4 w-4 mr-2" />
                    All Transactions
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefreshData}
                    disabled={isRefreshing}
                    className="fintech-button bg-transparent"
                  >
                    <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
                    {isRefreshing ? "Refreshing..." : "Refresh"}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadCSV}
                    disabled={isExporting}
                    className="fintech-button bg-transparent"
                  >
                    <Download className={`h-4 w-4 mr-2 ${isExporting ? "animate-spin" : ""}`} />
                    {isExporting ? "Exporting..." : "Export"}
                  </Button>
                </div>
              </div>
              {isRealTimeEnabled && (
                <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium text-green-800">Real-time monitoring active</span>
                  </div>
                  <span className="text-xs text-green-600">Last update: {lastUpdate.toLocaleTimeString()}</span>
                </div>
              )}
              <TransactionsTable
                transactions={transactions}
                onFilterChange={setCurrentFilter}
                currentFilter={currentFilter}
                onRefresh={handleRefreshData}
              />
            </section>
          </div>
        )

      case "alerts":
        return (
          <div className="space-y-8">
            <section>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">Fraud Alerts</h2>
                  <p className="text-muted-foreground">Active fraud alerts and notifications</p>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="fintech-button bg-transparent">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh Alerts
                  </Button>
                </div>
              </div>

              <div className="grid gap-4">
                <Card className="fintech-card border-l-4 border-l-destructive">
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4">
                        <div className="p-2 rounded-lg bg-destructive/10">
                          <AlertTriangle className="h-6 w-6 text-destructive" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-destructive mb-1">High Risk Transaction Detected</h3>
                          <p className="text-muted-foreground mb-2">
                            Transaction ID: TXN-789456123 flagged for unusual amount pattern
                          </p>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-4 w-4" />2 minutes ago
                            </span>
                            <span className="flex items-center gap-1">
                              <DollarSign className="h-4 w-4" />
                              $15,750.00
                            </span>
                          </div>
                        </div>
                      </div>
                      <Badge variant="destructive">Critical</Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card className="fintech-card border-l-4 border-l-amber-500">
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4">
                        <div className="p-2 rounded-lg bg-amber-500/10">
                          <AlertTriangle className="h-6 w-6 text-amber-500" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-amber-600 mb-1">Suspicious Velocity Pattern</h3>
                          <p className="text-muted-foreground mb-2">
                            Customer C770487 showing unusual transaction frequency
                          </p>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-4 w-4" />
                              15 minutes ago
                            </span>
                            <span className="flex items-center gap-1">
                              <TrendingUp className="h-4 w-4" />
                              12 transactions/hour
                            </span>
                          </div>
                        </div>
                      </div>
                      <Badge variant="secondary" className="bg-amber-500/10 text-amber-600">
                        Medium
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card className="fintech-card border-l-4 border-l-blue-500">
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4">
                        <div className="p-2 rounded-lg bg-blue-500/10">
                          <Users className="h-6 w-6 text-blue-500" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-blue-600 mb-1">New Destination Account</h3>
                          <p className="text-muted-foreground mb-2">First-time transfer to account M436030 detected</p>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-4 w-4" />1 hour ago
                            </span>
                            <span className="flex items-center gap-1">
                              <DollarSign className="h-4 w-4" />
                              $932.91
                            </span>
                          </div>
                        </div>
                      </div>
                      <Badge variant="outline" className="border-blue-500 text-blue-600">
                        Info
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </section>
          </div>
        )

      case "reports":
        return (
          <div className="space-y-8">
            <section>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">Fraud Reports</h2>
                  <p className="text-muted-foreground">Generate and download comprehensive fraud reports</p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadCSV}
                    disabled={isExporting}
                    className="fintech-button bg-transparent"
                  >
                    <Download className={`h-4 w-4 mr-2 ${isExporting ? "animate-spin" : ""}`} />
                    {isExporting ? "Generating..." : "Generate Report"}
                  </Button>
                </div>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <Card className="fintech-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <FileText className="h-5 w-5 text-primary" />
                      </div>
                      Daily Fraud Summary
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      Comprehensive daily fraud detection report with key metrics and trends.
                    </p>
                    <Button className="w-full fintech-button" onClick={downloadCSV} disabled={isExporting}>
                      <Download className="h-4 w-4 mr-2" />
                      Download Daily Report
                    </Button>
                  </CardContent>
                </Card>

                <Card className="fintech-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-secondary/10">
                        <BarChart3 className="h-5 w-5 text-secondary" />
                      </div>
                      Analytics Report
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      Detailed analytics with charts, patterns, and risk assessments.
                    </p>
                    <Button
                      className="w-full fintech-button bg-secondary hover:bg-secondary/90"
                      onClick={downloadCSV}
                      disabled={isExporting}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download Analytics
                    </Button>
                  </CardContent>
                </Card>

                <Card className="fintech-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-accent/10">
                        <AlertTriangle className="h-5 w-5 text-accent" />
                      </div>
                      Suspicious Activity Report
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      Detailed report of all flagged transactions and suspicious patterns.
                    </p>
                    <Button
                      className="w-full fintech-button bg-accent hover:bg-accent/90"
                      onClick={downloadCSV}
                      disabled={isExporting}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download SAR Report
                    </Button>
                  </CardContent>
                </Card>

                <Card className="fintech-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-success/10">
                        <Brain className="h-5 w-5 text-success" />
                      </div>
                      Model Performance Report
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      AI model accuracy, training metrics, and performance analysis.
                    </p>
                    <Button
                      className="w-full fintech-button bg-success hover:bg-success/90"
                      onClick={downloadCSV}
                      disabled={isExporting}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download Model Report
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </section>
          </div>
        )

      case "settings":
        return (
          <div className="space-y-8">
            <section>
              <div className="mb-6">
                <h2 className="text-2xl font-bold text-foreground">System Settings</h2>
                <p className="text-muted-foreground">Configure fraud detection system parameters</p>
              </div>

              <div className="grid gap-6">
                <Card className="fintech-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Brain className="h-5 w-5 text-primary" />
                      </div>
                      AI Model Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">Auto-retrain Model</h4>
                        <p className="text-sm text-muted-foreground">Automatically retrain model with new data</p>
                      </div>
                      <Button
                        variant={isRealTimeEnabled ? "default" : "outline"}
                        size="sm"
                        onClick={handleToggleRealTime}
                        className="fintech-button"
                      >
                        {isRealTimeEnabled ? "Enabled" : "Disabled"}
                      </Button>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">Risk Threshold</h4>
                        <p className="text-sm text-muted-foreground">Minimum risk score for flagging transactions</p>
                      </div>
                      <Badge variant="outline">0.75</Badge>
                    </div>
                    <Button className="w-full fintech-button" onClick={handleTrainModel} disabled={isTraining}>
                      <Brain className={`h-4 w-4 mr-2 ${isTraining ? "animate-spin" : ""}`} />
                      {isTraining ? "Training Model..." : "Retrain Model Now"}
                    </Button>
                  </CardContent>
                </Card>

                <Card className="fintech-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-secondary/10">
                        <Bell className="h-5 w-5 text-secondary" />
                      </div>
                      Alert Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">Real-time Alerts</h4>
                        <p className="text-sm text-muted-foreground">
                          Send immediate notifications for high-risk transactions
                        </p>
                      </div>
                      <Button variant="default" size="sm" className="fintech-button">
                        Enabled
                      </Button>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">Email Notifications</h4>
                        <p className="text-sm text-muted-foreground">Send daily summary reports via email</p>
                      </div>
                      <Button variant="outline" size="sm" className="fintech-button bg-transparent">
                        Disabled
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card className="fintech-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-accent/10">
                        <Activity className="h-5 w-5 text-accent" />
                      </div>
                      System Monitoring
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">Real-time Monitoring</h4>
                        <p className="text-sm text-muted-foreground">Monitor transactions in real-time</p>
                      </div>
                      <Button
                        variant={isRealTimeEnabled ? "default" : "outline"}
                        size="sm"
                        onClick={handleToggleRealTime}
                        className="fintech-button"
                      >
                        {isRealTimeEnabled ? "Active" : "Inactive"}
                      </Button>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">Data Refresh Rate</h4>
                        <p className="text-sm text-muted-foreground">Auto-refresh transactions every 30 seconds</p>
                      </div>
                      <Badge variant={isRealTimeEnabled ? "default" : "outline"}>
                        {isRealTimeEnabled ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                    <Button className="w-full fintech-button" onClick={() => setIsRealTimeEnabled(!isRealTimeEnabled)}>
                      <Activity className="h-4 w-4 mr-2" />
                      {isRealTimeEnabled ? "Disable Real-time" : "Enable Real-time"}
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </section>
          </div>
        )

      default:
        return null
    }
  }

  if (!analytics) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="text-center space-y-4">
          <div className="skeleton-card w-16 h-16 mx-auto rounded-full"></div>
          <div className="skeleton-text w-48 mx-auto"></div>
          <div className="skeleton-text w-32 mx-auto"></div>
        </div>
      </div>
    )
  }

  const handleRefresh = () => {
    handleRefreshData()
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 w-full border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="flex h-16 items-center px-6">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" className="md:hidden" onClick={() => setSidebarOpen(!sidebarOpen)}>
              {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg fintech-gradient">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-primary">Fraudet</h1>
                <p className="text-xs text-muted-foreground hidden sm:block">Fraud Detection System</p>
              </div>
            </div>
          </div>

          <div className="flex-1 flex items-center justify-center px-6">
            <div className="relative w-full max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search transactions..."
                className="w-full pl-10 pr-4 py-2 rounded-xl border border-border bg-background/50 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative">
              <Button
                variant="ghost"
                size="sm"
                className="relative p-2"
                onClick={() => setShowNotifications(!showNotifications)}
              >
                <Bell className="h-5 w-5" />
                {notifications.filter((n) => !n.read).length > 0 && (
                  <span className="absolute -top-1 -right-1 h-4 w-4 bg-destructive rounded-full text-xs text-white flex items-center justify-center">
                    {notifications.filter((n) => !n.read).length}
                  </span>
                )}
              </Button>

              {showNotifications && (
                <div className="absolute right-0 top-full mt-2 w-80 bg-background border border-border rounded-lg shadow-lg z-50">
                  <div className="p-4 border-b border-border">
                    <h3 className="font-semibold text-foreground">Notifications</h3>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {notifications.map((notification) => (
                      <div
                        key={notification.id}
                        className={`p-4 border-b border-border hover:bg-muted/50 cursor-pointer ${
                          !notification.read ? "bg-primary/5" : ""
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <div
                            className={`p-1 rounded-full ${
                              notification.type === "alert"
                                ? "bg-destructive/10"
                                : notification.type === "warning"
                                  ? "bg-amber-500/10"
                                  : "bg-primary/10"
                            }`}
                          >
                            {notification.type === "alert" && <AlertTriangle className="h-4 w-4 text-destructive" />}
                            {notification.type === "warning" && <AlertTriangle className="h-4 w-4 text-amber-500" />}
                            {notification.type === "info" && <Bell className="h-4 w-4 text-primary" />}
                          </div>
                          <div className="flex-1">
                            <h4 className="font-medium text-sm text-foreground">{notification.title}</h4>
                            <p className="text-sm text-muted-foreground mt-1">{notification.message}</p>
                            <p className="text-xs text-muted-foreground mt-2">{notification.time}</p>
                          </div>
                          {!notification.read && <div className="w-2 h-2 bg-primary rounded-full"></div>}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="p-3 border-t border-border">
                    <Button variant="ghost" size="sm" className="w-full text-sm">
                      Mark all as read
                    </Button>
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 pl-3 border-l">
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                <Shield className="h-4 w-4 text-primary" />
              </div>
              <div className="hidden sm:block">
                <p className="text-sm font-medium">Bank System</p>
                <p className="text-xs text-muted-foreground">Fraud Detection</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        <aside
          className={`fixed inset-y-0 left-0 z-40 w-64 bg-sidebar border-r border-sidebar-border transition-transform duration-300 ease-in-out ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0 md:static md:inset-0`}
        >
          <div className="flex flex-col h-full pt-16 md:pt-0">
            <div className="p-6 border-b border-sidebar-border">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg fintech-gradient">
                  <BarChart3 className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold text-sidebar-foreground">Dashboard</h2>
                  <p className="text-xs text-muted-foreground">Fraud Detection</p>
                </div>
              </div>
            </div>

            <nav className="flex-1 p-4 space-y-2">
              <Button
                variant="ghost"
                className={`w-full justify-start gap-3 ${activeView === "overview" ? "bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90" : "hover:bg-sidebar-accent/10 hover:text-sidebar-accent"}`}
                onClick={() => setActiveView("overview")}
              >
                <Home className="h-4 w-4" />
                Fraud Overview
              </Button>
              <Button
                variant="ghost"
                className={`w-full justify-start gap-3 ${activeView === "transactions" ? "bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90" : "hover:bg-sidebar-accent/10 hover:text-sidebar-accent"}`}
                onClick={() => setActiveView("transactions")}
              >
                <CreditCard className="h-4 w-4" />
                Transactions
              </Button>
              <Button
                variant="ghost"
                className={`w-full justify-start gap-3 ${activeView === "alerts" ? "bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90" : "hover:bg-sidebar-accent/10 hover:text-sidebar-accent"}`}
                onClick={() => setActiveView("alerts")}
              >
                <AlertTriangle className="h-4 w-4" />
                Alerts
                <Badge className="ml-auto bg-destructive text-destructive-foreground">3</Badge>
              </Button>
              <Button
                variant="ghost"
                className={`w-full justify-start gap-3 ${activeView === "reports" ? "bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90" : "hover:bg-sidebar-accent/10 hover:text-sidebar-accent"}`}
                onClick={() => setActiveView("reports")}
              >
                <FileText className="h-4 w-4" />
                Reports
              </Button>
              <Button
                variant="ghost"
                className={`w-full justify-start gap-3 ${activeView === "settings" ? "bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90" : "hover:bg-sidebar-accent/10 hover:text-sidebar-accent"}`}
                onClick={() => setActiveView("settings")}
              >
                <Settings className="h-4 w-4" />
                Settings
              </Button>
            </nav>

            <div className="p-4 border-t border-sidebar-border">
              <div className="flex items-center gap-3 p-3 rounded-xl bg-sidebar-accent/10">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                <span className="font-medium">System Status</span>
                <span className="text-white/80">All systems operational</span>
              </div>
            </div>
          </div>
        </aside>

        <main className="flex-1 md:ml-0">
          <div className="p-6">{renderContent()}</div>
        </main>
      </div>

      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-30 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}
    </div>
  )
}
