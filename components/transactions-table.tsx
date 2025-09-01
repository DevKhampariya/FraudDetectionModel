"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Download, Eye, X, MapPin, CreditCard, Shield, Clock, User, Building2, Globe, Smartphone } from "lucide-react"
import { useState } from "react"

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
}

interface TransactionsTableProps {
  transactions: Transaction[]
  onFilterChange: (filter: string) => void
  currentFilter: string
  onRefresh?: () => void
}

export function TransactionsTable({ transactions, onFilterChange, currentFilter, onRefresh }: TransactionsTableProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(amount)
  }

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const getStatusBadge = (status: string, riskScore: number) => {
    if (status === "suspicious") {
      return <Badge variant="destructive">Suspicious</Badge>
    } else if (status === "normal" && riskScore > 50) {
      return <Badge variant="secondary">Normal</Badge>
    } else {
      return <Badge variant="outline">Normal</Badge>
    }
  }

  const getRiskColor = (riskScore: number) => {
    if (riskScore >= 80) return "text-red-600 font-semibold"
    if (riskScore >= 60) return "text-orange-600 font-medium"
    if (riskScore >= 40) return "text-yellow-600"
    return "text-green-600"
  }

  const downloadCSV = () => {
    const headers = [
      "Transaction ID",
      "Customer ID",
      "Type",
      "Amount",
      "Risk Score",
      "Timestamp",
      "Status",
      "Reason",
      "Location",
      "Merchant",
      "Category",
      "Payment Method",
      "Device Info",
      "IP Address",
      "Country",
      "City",
    ]
    const csvContent = [
      headers.join(","),
      ...transactions.map((t) =>
        [
          t.transactionId,
          t.customerId,
          t.type,
          t.amount,
          t.riskScore,
          t.timestamp,
          t.status,
          `"${t.reason}"`,
          t.location,
          t.merchant,
          t.category,
          t.paymentMethod,
          t.deviceInfo,
          t.ipAddress,
          t.country,
          t.city,
        ].join(","),
      ),
    ].join("\n")

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `transactions_${currentFilter}_${new Date().toISOString().split("T")[0]}.csv`
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null)
  const [showDetails, setShowDetails] = useState(false)

  const viewTransactionDetails = (transaction: Transaction) => {
    setSelectedTransaction(transaction)
    setShowDetails(true)
  }

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Transaction Monitor</CardTitle>
            <div className="flex items-center gap-2">
              <Button onClick={onRefresh} variant="default" size="sm" disabled={!onRefresh}>
                Refresh
              </Button>
              <Select value={currentFilter} onValueChange={onFilterChange}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Filter transactions" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Transactions</SelectItem>
                  <SelectItem value="suspicious">Suspicious Only</SelectItem>
                  <SelectItem value="normal">Normal Only</SelectItem>
                  <SelectItem value="today">Today</SelectItem>
                  <SelectItem value="week">This Week</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={downloadCSV} variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export CSV
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Transaction ID</th>
                  <th className="text-left p-2">Customer</th>
                  <th className="text-left p-2">Merchant</th>
                  <th className="text-left p-2">Location</th>
                  <th className="text-left p-2">Type</th>
                  <th className="text-right p-2">Amount</th>
                  <th className="text-center p-2">Risk Score</th>
                  <th className="text-left p-2">Status</th>
                  <th className="text-left p-2">Timestamp</th>
                  <th className="text-center p-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((transaction) => (
                  <tr key={transaction.transactionId} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-mono text-sm">{transaction.transactionId}</td>
                    <td className="p-2">{transaction.customerId}</td>
                    <td className="p-2">{transaction.merchant}</td>
                    <td className="p-2 text-sm">
                      {transaction.city}, {transaction.country}
                    </td>
                    <td className="p-2 capitalize">{transaction.type}</td>
                    <td className="p-2 text-right font-medium">{formatCurrency(transaction.amount)}</td>
                    <td className="p-2 text-center">
                      <span className={getRiskColor(transaction.riskScore)}>{transaction.riskScore}</span>
                    </td>
                    <td className="p-2">{getStatusBadge(transaction.status, transaction.riskScore)}</td>
                    <td className="p-2 text-sm text-muted-foreground">{formatDate(transaction.timestamp)}</td>
                    <td className="p-2 text-center">
                      <Button variant="ghost" size="sm" onClick={() => viewTransactionDetails(transaction)}>
                        <Eye className="h-4 w-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {transactions.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                No transactions found for the selected filter.
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {showDetails && selectedTransaction && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-4xl max-h-[90vh] overflow-hidden fintech-card shadow-2xl border-0">
            <CardHeader className="bg-gradient-to-r from-primary/10 via-secondary/10 to-accent/10 border-b">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-primary/20">
                    <Eye className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <CardTitle className="text-xl">Transaction Details</CardTitle>
                    <p className="text-sm text-muted-foreground">Complete transaction information</p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowDetails(false)}
                  className="hover:bg-destructive/10 hover:text-destructive rounded-full h-8 w-8 p-0"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
              <div className="space-y-6">
                {/* Transaction Overview */}
                <div className="bg-gradient-to-r from-primary/5 to-secondary/5 rounded-xl p-4 border border-primary/20">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <CreditCard className="h-5 w-5 text-primary" />
                    Transaction Overview
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center p-3 bg-white/50 rounded-lg">
                      <p className="text-2xl font-bold text-primary">{formatCurrency(selectedTransaction.amount)}</p>
                      <p className="text-sm text-muted-foreground">Amount</p>
                    </div>
                    <div className="text-center p-3 bg-white/50 rounded-lg">
                      <p className={`text-2xl font-bold ${getRiskColor(selectedTransaction.riskScore)}`}>
                        {selectedTransaction.riskScore}
                      </p>
                      <p className="text-sm text-muted-foreground">Risk Score</p>
                    </div>
                    <div className="text-center p-3 bg-white/50 rounded-lg">
                      {getStatusBadge(selectedTransaction.status, selectedTransaction.riskScore)}
                      <p className="text-sm text-muted-foreground mt-1">Status</p>
                    </div>
                  </div>
                </div>

                {/* Transaction Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="fintech-card border-l-4 border-l-primary">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base flex items-center gap-2">
                        <User className="h-4 w-4 text-primary" />
                        Customer Information
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">Transaction ID</span>
                        <span className="font-mono text-sm bg-muted/50 px-2 py-1 rounded">
                          {selectedTransaction.transactionId}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">Customer ID</span>
                        <span className="font-medium">{selectedTransaction.customerId}</span>
                      </div>
                      <div className="flex justify-between items-center py-2">
                        <span className="text-sm text-muted-foreground">Transaction Type</span>
                        <Badge variant="outline" className="capitalize">
                          {selectedTransaction.type}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="fintech-card border-l-4 border-l-secondary">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base flex items-center gap-2">
                        <Building2 className="h-4 w-4 text-secondary" />
                        Merchant Details
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">Merchant</span>
                        <span className="font-medium">{selectedTransaction.merchant}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">Category</span>
                        <Badge variant="secondary">{selectedTransaction.category}</Badge>
                      </div>
                      <div className="flex justify-between items-center py-2">
                        <span className="text-sm text-muted-foreground">Payment Method</span>
                        <span className="font-medium">{selectedTransaction.paymentMethod}</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="fintech-card border-l-4 border-l-accent">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base flex items-center gap-2">
                        <MapPin className="h-4 w-4 text-accent" />
                        Location Information
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">Location</span>
                        <span className="font-medium">{selectedTransaction.location}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">City</span>
                        <span className="font-medium">{selectedTransaction.city}</span>
                      </div>
                      <div className="flex justify-between items-center py-2">
                        <span className="text-sm text-muted-foreground">Country</span>
                        <div className="flex items-center gap-2">
                          <Globe className="h-3 w-3 text-muted-foreground" />
                          <span className="font-medium">{selectedTransaction.country}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="fintech-card border-l-4 border-l-success">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base flex items-center gap-2">
                        <Smartphone className="h-4 w-4 text-success" />
                        Technical Details
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">Device Info</span>
                        <span className="font-medium text-sm">{selectedTransaction.deviceInfo}</span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-muted/30">
                        <span className="text-sm text-muted-foreground">IP Address</span>
                        <span className="font-mono text-sm bg-muted/50 px-2 py-1 rounded">
                          {selectedTransaction.ipAddress}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-2">
                        <span className="text-sm text-muted-foreground">Timestamp</span>
                        <div className="flex items-center gap-2">
                          <Clock className="h-3 w-3 text-muted-foreground" />
                          <span className="font-medium text-sm">{formatDate(selectedTransaction.timestamp)}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Risk Analysis */}
                <Card className="fintech-card border-l-4 border-l-warning">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Shield className="h-4 w-4 text-warning" />
                      Risk Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gradient-to-r from-warning/10 to-warning/5 p-4 rounded-lg border border-warning/20">
                      <p className="text-sm leading-relaxed">{selectedTransaction.reason}</p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  )
}
