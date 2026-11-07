using System;

namespace Shop.Payments
{
    public class PaymentService
    {
        public decimal Charge(decimal amount)
        {
            if (amount <= 0) throw new ArgumentException("amount");
            return amount;
        }

        public void Refund(string paymentId)
        {
            Console.WriteLine($"Refund {paymentId}");
        }
    }
}
