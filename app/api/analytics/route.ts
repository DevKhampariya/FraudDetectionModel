import { NextResponse } from "next/server"

interface Transaction {
  step: number
  type: string
  amount: number
  nameOrig: string
  oldbalanceOrg: number
  newbalanceOrig: number
  nameDest: string
  oldbalanceDest: number
  newbalanceDest: number
  hour: number
  day_of_week: number
  is_weekend: number
  is_night: number
  avg_amount: number
  transaction_count: number
  amount_std: number
  amount_to_balance_ratio: number
  amount_zscore: number
  is_round_amount: number
  balance_drained: number
  dest_transaction_count: number
  is_new_dest: number
  hourly_txn_count: number
  balance_change_ratio: number
  unique_destinations: number
  unique_senders_to_dest: number
  deviation_flag: number
  high_velocity_flag: number
  suspicious_time_flag: number
  txn_frequency_per_day: number
  amount_risk_score: number
  behavioral_risk_score: number
  isFraud: number
}

// Cache for CSV data to avoid repeated fetches
let cachedTransactions: Transaction[] | null = null
let lastFetchTime = 0
const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes

async function fetchTransactionsFromCSV(): Promise<Transaction[]> {
  const now = Date.now()

  // Return cached data if still valid
  if (cachedTransactions && now - lastFetchTime < CACHE_DURATION) {
    return cachedTransactions
  }

  try {
    console.log("[v0] Fetching CSV data for analytics...")
    const csvUrl =
      "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/enhanced_fraud_dataset-ZMjbsPepn5xUXRFrSs12xgofaGgWk1.csv"
    const response = await fetch(csvUrl)

    if (!response.ok) {
      throw new Error(`Failed to fetch CSV: ${response.statusText}`)
    }

    const csvText = await response.text()
    const lines = csvText.trim().split(/\r?\n/)

    if (lines.length < 2) {
      throw new Error("CSV file appears to be empty or invalid")
    }

    const headers = lines[0].split(",").map((h) => h.replace(/"/g, "").trim())
    console.log("[v0] CSV headers:", headers)

    const transactions: Transaction[] = []

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim()
      if (!line) continue

      const values = []
      let current = ""
      let inQuotes = false

      for (let j = 0; j < line.length; j++) {
        const char = line[j]
        if (char === '"') {
          inQuotes = !inQuotes
        } else if (char === "," && !inQuotes) {
          values.push(current.trim())
          current = ""
        } else {
          current += char
        }
      }
      values.push(current.trim())

      if (values.length >= 32) {
        try {
          transactions.push({
            step: Number.parseInt(values[0]) || 0,
            type: values[1] || "",
            amount: Number.parseFloat(values[2]) || 0,
            nameOrig: values[3] || "",
            oldbalanceOrg: Number.parseFloat(values[4]) || 0,
            newbalanceOrig: Number.parseFloat(values[5]) || 0,
            nameDest: values[6] || "",
            oldbalanceDest: Number.parseFloat(values[7]) || 0,
            newbalanceDest: Number.parseFloat(values[8]) || 0,
            hour: Number.parseInt(values[9]) || 0,
            day_of_week: Number.parseInt(values[10]) || 0,
            is_weekend: Number.parseInt(values[11]) || 0,
            is_night: Number.parseInt(values[12]) || 0,
            avg_amount: Number.parseFloat(values[13]) || 0,
            transaction_count: Number.parseInt(values[14]) || 0,
            amount_std: Number.parseFloat(values[15]) || 0,
            amount_to_balance_ratio: Number.parseFloat(values[16]) || 0,
            amount_zscore: Number.parseFloat(values[17]) || 0,
            is_round_amount: Number.parseInt(values[18]) || 0,
            balance_drained: Number.parseInt(values[19]) || 0,
            dest_transaction_count: Number.parseInt(values[20]) || 0,
            is_new_dest: Number.parseInt(values[21]) || 0,
            hourly_txn_count: Number.parseInt(values[22]) || 0,
            balance_change_ratio: Number.parseFloat(values[23]) || 0,
            unique_destinations: Number.parseInt(values[24]) || 0,
            unique_senders_to_dest: Number.parseInt(values[25]) || 0,
            deviation_flag: Number.parseInt(values[26]) || 0,
            high_velocity_flag: Number.parseInt(values[27]) || 0,
            suspicious_time_flag: Number.parseInt(values[28]) || 0,
            txn_frequency_per_day: Number.parseFloat(values[29]) || 0,
            amount_risk_score: Number.parseFloat(values[30]) || 0,
            behavioral_risk_score: Number.parseFloat(values[31]) || 0,
            isFraud: Number.parseInt(values[32]) || 0,
          })
        } catch (error) {
          console.error(`[v0] Error parsing row ${i}:`, error)
          continue
        }
      }
    }

    console.log(`[v0] Total rows: ${lines.length - 1}`)
    console.log(`[v0] Successfully parsed transactions: ${transactions.length}`)

    if (transactions.length > 0) {
      console.log("[v0] Sample transaction:", transactions[0])
      const fraudCount = transactions.filter((t) => t.isFraud === 1).length
      const fraudRate = (fraudCount / transactions.length) * 100
      console.log(`[v0] Fraud rate from CSV: ${fraudRate.toFixed(2)}%`)
      console.log(`[v0] Total fraudulent transactions: ${fraudCount}`)
      console.log(`[v0] Total legitimate transactions: ${transactions.length - fraudCount}`)
    }

    // Cache the results
    cachedTransactions = transactions
    lastFetchTime = now

    return transactions
  } catch (error) {
    console.error("[v0] Error fetching CSV data for analytics:", error)
    return []
  }
}

function calculateAnalytics(transactions: Transaction[]) {
  const totalTransactions = transactions.length
  const suspiciousTransactions = transactions.filter((t) => t.isFraud === 1).length
  const fraudRate = totalTransactions > 0 ? (suspiciousTransactions / totalTransactions) * 100 : 0

  const totalAmount = transactions.reduce((sum, t) => sum + t.amount, 0)
  const suspiciousAmount = transactions.filter((t) => t.isFraud === 1).reduce((sum, t) => sum + t.amount, 0)

  const dailyStats = []
  for (let i = 6; i >= 0; i--) {
    const date = new Date()
    date.setDate(date.getDate() - i)
    const dateStr = date.toISOString().split("T")[0]

    const dayTransactions = transactions.filter((t, index) => {
      const dayIndex = Math.floor((t.step / totalTransactions) * 7)
      return dayIndex === i
    })

    const daySuspicious = dayTransactions.filter((t) => t.isFraud === 1).length
    const dayFraudRate = dayTransactions.length > 0 ? (daySuspicious / dayTransactions.length) * 100 : 0

    dailyStats.push({
      date: dateStr,
      total: dayTransactions.length,
      suspicious: daySuspicious,
      fraudRate: Math.round(dayFraudRate * 10) / 10,
    })
  }

  const highRisk = transactions.filter((t) => t.amount_risk_score >= 2.0).length
  const mediumRisk = transactions.filter((t) => t.amount_risk_score >= 1.0 && t.amount_risk_score < 2.0).length
  const lowRisk = transactions.filter((t) => t.amount_risk_score < 1.0).length

  const riskDistribution = [
    {
      risk: "High (≥2.0)",
      count: highRisk,
      percentage: totalTransactions > 0 ? Math.round((highRisk / totalTransactions) * 1000) / 10 : 0,
    },
    {
      risk: "Medium (1.0-2.0)",
      count: mediumRisk,
      percentage: totalTransactions > 0 ? Math.round((mediumRisk / totalTransactions) * 1000) / 10 : 0,
    },
    {
      risk: "Low (<1.0)",
      count: lowRisk,
      percentage: totalTransactions > 0 ? Math.round((lowRisk / totalTransactions) * 1000) / 10 : 0,
    },
  ]

  const typeGroups = transactions.reduce(
    (acc, t) => {
      if (!acc[t.type]) {
        acc[t.type] = { total: 0, suspicious: 0 }
      }
      acc[t.type].total++
      if (t.isFraud === 1) {
        acc[t.type].suspicious++
      }
      return acc
    },
    {} as Record<string, { total: number; suspicious: number }>,
  )

  const transactionTypes = Object.entries(typeGroups).map(([type, data]) => ({
    type,
    total: data.total,
    suspicious: data.suspicious,
  }))

  return {
    totalTransactions,
    suspiciousTransactions,
    fraudRate: Math.round(fraudRate * 10) / 10,
    totalAmount: Math.round(totalAmount * 100) / 100,
    suspiciousAmount: Math.round(suspiciousAmount * 100) / 100,
    dailyStats,
    riskDistribution,
    transactionTypes,
  }
}

export async function GET() {
  try {
    console.log("[v0] Analytics API called")

    const transactions = await fetchTransactionsFromCSV()

    if (transactions.length === 0) {
      return NextResponse.json({
        totalTransactions: 0,
        suspiciousTransactions: 0,
        fraudRate: 0,
        totalAmount: 0,
        suspiciousAmount: 0,
        dailyStats: [],
        riskDistribution: [],
        transactionTypes: [],
        error: "No transaction data available",
      })
    }

    const analytics = calculateAnalytics(transactions)

    console.log("[v0] Analytics calculated:", {
      total: analytics.totalTransactions,
      suspicious: analytics.suspiciousTransactions,
      fraudRate: analytics.fraudRate,
    })

    return NextResponse.json(analytics)
  } catch (error) {
    console.error("[v0] Error fetching analytics:", error)
    return NextResponse.json({ error: "Failed to fetch analytics" }, { status: 500 })
  }
}
