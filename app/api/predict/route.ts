import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { amount, type, customerId, timestamp } = body

    // Mock ML prediction logic
    // In a real app, this would use your trained model
    const baseRisk = Math.random() * 100

    // Adjust risk based on transaction characteristics
    let riskScore = baseRisk

    // High amounts increase risk
    if (amount > 5000) riskScore += 20
    if (amount > 10000) riskScore += 30

    // Certain transaction types are riskier
    if (type === "withdrawal") riskScore += 15
    if (type === "transfer") riskScore += 10

    // Time-based risk (late night transactions)
    const hour = new Date(timestamp).getHours()
    if (hour < 6 || hour > 22) riskScore += 25

    // Cap at 100
    riskScore = Math.min(riskScore, 100)

    const status = riskScore > 70 ? "suspicious" : "normal"
    const reason =
      riskScore > 70 ? "High risk score detected based on amount, type, and timing" : "Normal transaction pattern"

    const prediction = {
      transactionId: `txn_${Math.random().toString(36).substr(2, 9)}`,
      customerId,
      type,
      amount,
      riskScore: Math.round(riskScore),
      timestamp,
      status,
      reason,
      confidence: Math.random() * 0.3 + 0.7, // 70-100% confidence
    }

    return NextResponse.json(prediction)
  } catch (error) {
    console.error("Error making prediction:", error)
    return NextResponse.json({ error: "Failed to make prediction" }, { status: 500 })
  }
}
