import { type NextRequest, NextResponse } from "next/server"

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
  // Computed fields for UI compatibility
  transactionId: string
  customerId: string
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

// Cache for CSV data to avoid repeated fetches
let cachedTransactions: Transaction[] | null = null
let lastFetchTime = 0
const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ""
  let inQuotes = false
  let i = 0

  while (i < line.length) {
    const char = line[i]
    const nextChar = line[i + 1]

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        // Escaped quote
        current += '"'
        i += 2
        continue
      } else {
        // Toggle quote state
        inQuotes = !inQuotes
      }
    } else if (char === "," && !inQuotes) {
      // End of field
      result.push(current.trim())
      current = ""
      i++
      continue
    } else {
      current += char
    }
    i++
  }

  // Add the last field
  result.push(current.trim())
  return result
}

function generateMerchantName(type: string, amount: number): string {
  const merchants = {
    PAYMENT: [
      "Amazon.com",
      "Walmart",
      "Target",
      "Best Buy",
      "Home Depot",
      "Costco",
      "McDonald's",
      "Starbucks",
      "Shell Gas Station",
      "Chevron",
      "Uber",
      "Lyft",
      "Netflix",
      "Spotify",
      "Apple Store",
      "Google Play",
    ],
    TRANSFER: ["Bank of America", "Chase Bank", "Wells Fargo", "Citibank", "PayPal", "Venmo", "Zelle", "Western Union"],
    CASH_OUT: ["ATM Withdrawal", "Bank Branch", "Credit Union", "Cash Advance", "Payday Loan"],
    DEBIT: ["Direct Debit", "Auto Payment", "Subscription Service", "Utility Payment", "Insurance Payment"],
  }

  const typeKey = type as keyof typeof merchants
  const merchantList = merchants[typeKey] || merchants.PAYMENT
  return merchantList[Math.floor(Math.random() * merchantList.length)]
}

function generateLocation(hour: number, is_weekend: number): string {
  const locations = is_weekend
    ? ["Shopping Mall", "Restaurant District", "Entertainment Center", "Retail Park", "Downtown"]
    : ["Business District", "Office Complex", "Commercial Area", "City Center", "Financial District"]

  return locations[Math.floor(Math.random() * locations.length)]
}

function generateCategory(type: string): string {
  const categories = {
    PAYMENT: ["Retail", "Food & Dining", "Gas & Fuel", "Entertainment", "Shopping", "Groceries"],
    TRANSFER: ["Banking", "P2P Transfer", "Investment", "Savings", "Loan Payment"],
    CASH_OUT: ["ATM", "Cash Advance", "Banking", "Financial Services"],
    DEBIT: ["Utilities", "Insurance", "Subscription", "Auto Payment", "Bills"],
  }

  const typeKey = type as keyof typeof categories
  const categoryList = categories[typeKey] || categories.PAYMENT
  return categoryList[Math.floor(Math.random() * categoryList.length)]
}

function generatePaymentMethod(type: string): string {
  const methods = {
    PAYMENT: ["Credit Card", "Debit Card", "Mobile Wallet", "Contactless", "Chip Card"],
    TRANSFER: ["Online Banking", "Mobile App", "Wire Transfer", "ACH Transfer"],
    CASH_OUT: ["ATM Card", "Debit Card", "Bank Card"],
    DEBIT: ["Auto Debit", "Direct Debit", "Online Payment", "Bank Transfer"],
  }

  const typeKey = type as keyof typeof methods
  const methodList = methods[typeKey] || methods.PAYMENT
  return methodList[Math.floor(Math.random() * methodList.length)]
}

function generateDeviceInfo(is_night: number, suspicious_time_flag: number): string {
  const devices = suspicious_time_flag
    ? ["Unknown Device", "Unregistered Mobile", "New Browser", "VPN Connection"]
    : ["iPhone 14", "Samsung Galaxy", "Chrome Browser", "Safari Browser", "Android App", "iOS App"]

  return devices[Math.floor(Math.random() * devices.length)]
}

function generateIPAddress(): string {
  return `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`
}

function generateCountryCity(suspicious_time_flag: number): { country: string; city: string } {
  const locations = suspicious_time_flag
    ? [
        { country: "Unknown", city: "Unknown" },
        { country: "Nigeria", city: "Lagos" },
        { country: "Russia", city: "Moscow" },
        { country: "China", city: "Beijing" },
      ]
    : [
        { country: "United States", city: "New York" },
        { country: "United States", city: "Los Angeles" },
        { country: "United States", city: "Chicago" },
        { country: "United States", city: "Houston" },
        { country: "United States", city: "Phoenix" },
        { country: "Canada", city: "Toronto" },
        { country: "Canada", city: "Vancouver" },
      ]

  return locations[Math.floor(Math.random() * locations.length)]
}

async function fetchTransactionsFromCSV(): Promise<Transaction[]> {
  const now = Date.now()

  // Return cached data if still valid
  if (cachedTransactions && now - lastFetchTime < CACHE_DURATION) {
    return cachedTransactions
  }

  try {
    console.log("[v0] Fetching CSV data from URL...")
    const csvUrl =
      "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/enhanced_fraud_dataset-ZMjbsPepn5xUXRFrSs12xgofaGgWk1.csv"
    const response = await fetch(csvUrl)

    if (!response.ok) {
      throw new Error(`Failed to fetch CSV: ${response.statusText}`)
    }

    const csvText = await response.text()
    const lines = csvText.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim().split("\n")
    const headers = parseCSVLine(lines[0])

    console.log("[v0] CSV headers:", headers)
    console.log("[v0] Total rows:", lines.length - 1)

    const transactions: Transaction[] = []

    for (let i = 1; i < lines.length; i++) {
      if (lines[i].trim() === "") continue // Skip empty lines

      const values = parseCSVLine(lines[i])

      if (values.length >= 32) {
        // Expecting 32 columns based on schema
        // Parse all the enhanced features
        const step = Number.parseInt(values[0] || "0")
        const type = values[1]?.replace(/"/g, "") || "DEBIT"
        const amount = Number.parseFloat(values[2] || "0")
        const nameOrig = values[3]?.replace(/"/g, "") || ""
        const oldbalanceOrg = Number.parseFloat(values[4] || "0")
        const newbalanceOrig = Number.parseFloat(values[5] || "0")
        const nameDest = values[6]?.replace(/"/g, "") || ""
        const oldbalanceDest = Number.parseFloat(values[7] || "0")
        const newbalanceDest = Number.parseFloat(values[8] || "0")
        const hour = Number.parseInt(values[9] || "0")
        const day_of_week = Number.parseInt(values[10] || "0")
        const is_weekend = Number.parseInt(values[11] || "0")
        const is_night = Number.parseInt(values[12] || "0")
        const avg_amount = Number.parseFloat(values[13] || "0")
        const transaction_count = Number.parseInt(values[14] || "0")
        const amount_std = Number.parseFloat(values[15] || "0")
        const amount_to_balance_ratio = Number.parseFloat(values[16] || "0")
        const amount_zscore = Number.parseFloat(values[17] || "0")
        const is_round_amount = Number.parseInt(values[18] || "0")
        const balance_drained = Number.parseInt(values[19] || "0")
        const dest_transaction_count = Number.parseInt(values[20] || "0")
        const is_new_dest = Number.parseInt(values[21] || "0")
        const hourly_txn_count = Number.parseInt(values[22] || "0")
        const balance_change_ratio = Number.parseFloat(values[23] || "0")
        const unique_destinations = Number.parseInt(values[24] || "0")
        const unique_senders_to_dest = Number.parseInt(values[25] || "0")
        const deviation_flag = Number.parseInt(values[26] || "0")
        const high_velocity_flag = Number.parseInt(values[27] || "0")
        const suspicious_time_flag = Number.parseInt(values[28] || "0")
        const txn_frequency_per_day = Number.parseFloat(values[29] || "0")
        const amount_risk_score = Number.parseFloat(values[30] || "0")
        const behavioral_risk_score = Number.parseFloat(values[31] || "0")
        const isFraud = Number.parseInt(values[32] || "0")

        const locationCity = generateCountryCity(suspicious_time_flag)

        const transaction: Transaction = {
          step,
          type,
          amount,
          nameOrig,
          oldbalanceOrg,
          newbalanceOrig,
          nameDest,
          oldbalanceDest,
          newbalanceDest,
          hour,
          day_of_week,
          is_weekend,
          is_night,
          avg_amount,
          transaction_count,
          amount_std,
          amount_to_balance_ratio,
          amount_zscore,
          is_round_amount,
          balance_drained,
          dest_transaction_count,
          is_new_dest,
          hourly_txn_count,
          balance_change_ratio,
          unique_destinations,
          unique_senders_to_dest,
          deviation_flag,
          high_velocity_flag,
          suspicious_time_flag,
          txn_frequency_per_day,
          amount_risk_score,
          behavioral_risk_score,
          isFraud,
          // Computed fields for UI
          transactionId: `txn_${step}_${nameOrig}`,
          customerId: nameOrig,
          riskScore: Math.round((amount_risk_score + behavioral_risk_score) * 50), // Scale to 0-100
          timestamp: new Date(Date.now() - (10000 - step) * 60000).toISOString(), // Simulate timestamps
          status: isFraud === 1 ? "suspicious" : "approved",
          reason:
            isFraud === 1
              ? `High risk: Amount risk ${amount_risk_score.toFixed(2)}, Behavioral risk ${behavioral_risk_score}`
              : "Normal transaction pattern",
          location: generateLocation(hour, is_weekend),
          merchant: generateMerchantName(type, amount),
          category: generateCategory(type),
          paymentMethod: generatePaymentMethod(type),
          deviceInfo: generateDeviceInfo(is_night, suspicious_time_flag),
          ipAddress: generateIPAddress(),
          country: locationCity.country,
          city: locationCity.city,
        }

        transactions.push(transaction)
      }
    }

    console.log("[v0] Successfully parsed transactions:", transactions.length)
    console.log("[v0] Sample transaction:", transactions[0])

    const fraudCount = transactions.filter((t) => t.isFraud === 1).length
    const fraudRate = transactions.length > 0 ? (fraudCount / transactions.length) * 100 : 0
    console.log("[v0] Fraud rate from CSV:", fraudRate.toFixed(2) + "%")
    console.log("[v0] Total fraudulent transactions:", fraudCount)
    console.log("[v0] Total legitimate transactions:", transactions.length - fraudCount)

    // Cache the results
    cachedTransactions = transactions
    lastFetchTime = now

    return transactions
  } catch (error) {
    console.error("[v0] Error fetching CSV data:", error)
    // Return empty array on error
    return []
  }
}

export const dynamic = "force-dynamic"
export const revalidate = 0

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const filter = searchParams.get("filter") || "all"
    const limit = Number.parseInt(searchParams.get("limit") || "50")
    const refresh = searchParams.get("refresh") || "0"

    console.log("[v0] API request - filter:", filter, "limit:", limit, "refresh:", refresh)

    // Fetch real transactions from CSV
    let transactions = await fetchTransactionsFromCSV()

    if (transactions.length === 0) {
      return NextResponse.json(
        {
          error: "No transaction data available",
          transactions: [],
          total: 0,
          filter,
        },
        {
          status: 404,
          headers: {
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            Pragma: "no-cache",
            Expires: "0",
          },
        },
      )
    }

    const refreshSeed = Number.parseInt(refresh) || 0
    const shuffledTransactions = [...transactions]

    // Simple shuffle algorithm using refresh count as seed
    for (let i = shuffledTransactions.length - 1; i > 0; i--) {
      const j = Math.floor(((((refreshSeed + i) * 9301 + 49297) % 233280) / 233280) * (i + 1))
      ;[shuffledTransactions[i], shuffledTransactions[j]] = [shuffledTransactions[j], shuffledTransactions[i]]
    }

    transactions = shuffledTransactions

    function clamp(n: number, min: number, max: number) {
      return Math.max(min, Math.min(max, n))
    }
    function seededDelta(index: number, seed: number) {
      // deterministic pseudo-random in [-10, +10]
      const r = ((((seed + index) * 9301 + 49297) % 233280) / 233280) * 2 - 1 // [-1, 1)
      return Math.round(r * 10)
    }
    transactions = transactions.map((t, i) => {
      const delta = seededDelta(i, refreshSeed)
      const newRisk = clamp((t.riskScore ?? 0) + delta, 0, 100)
      const newStatus = newRisk >= 80 ? "suspicious" : "approved"
      const newReason =
        newStatus === "suspicious"
          ? `High risk after refresh ${refreshSeed}: risk score ${newRisk}`
          : "Normal transaction pattern"

      return {
        ...t,
        riskScore: newRisk,
        status: newStatus,
        reason: newReason,
      }
    })

    // Apply filters
    if (filter === "suspicious") {
      transactions = transactions.filter((t) => t.status === "suspicious")
    } else if (filter === "normal") {
      transactions = transactions.filter((t) => t.riskScore > 50 && t.status !== "suspicious")
    } else if (filter === "today") {
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      transactions = transactions.filter((t) => new Date(t.timestamp) >= today)
    } else if (filter === "week") {
      const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
      transactions = transactions.filter((t) => new Date(t.timestamp) >= weekAgo)
    }

    // Sort by timestamp (newest first)
    transactions.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())

    // Limit results
    const limitedTransactions = transactions.slice(0, limit)

    console.log("[v0] Returning transactions:", limitedTransactions.length)

    return NextResponse.json(
      {
        transactions: limitedTransactions,
        total: transactions.length,
        filter,
      },
      {
        headers: {
          "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
          Pragma: "no-cache",
          Expires: "0",
        },
      },
    )
  } catch (error) {
    console.error("[v0] Error in transactions API:", error)
    return NextResponse.json(
      { error: "Failed to fetch transactions" },
      {
        status: 500,
        headers: {
          "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
          Pragma: "no-cache",
          Expires: "0",
        },
      },
    )
  }
}
