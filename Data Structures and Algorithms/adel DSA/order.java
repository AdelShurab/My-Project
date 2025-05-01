public class order {
    static class Order {
        int orderID;
        private DLLItems itemList;
        private double totalPrice;

        public Order(int orderID, DLLItems itemList) {
            this.orderID = orderID;
            this.itemList = itemList;
            this.totalPrice = 0.0;
        }

        public int getOrderID() {
            return this.orderID;
        }

        public void calculateTotalPrice() {
            if(itemList.head == null) {
                System.out.println("No items in the order!");
                return;
            }

            NodeItem temp = itemList.head;
            while(temp != null) {
                totalPrice += temp.value.getPrice();
                temp = temp.next;
            }
        }

        public void printOrder() {
            NodeItem temp = itemList.head;
            while(temp != null) {
                System.out.println("Item: " + temp.value.getDescription() + " Price: " + temp.value.getPrice());
                temp = temp.next;
            }
            System.out.println("Total Price: " + totalPrice);
        }
    }

}
