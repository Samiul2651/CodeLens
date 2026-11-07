import { formatPrice } from "./format.js";

export class Cart {
  constructor() {
    this.items = [];
  }

  add(item) {
    this.items.push(item);
  }

  total() {
    return this.items.reduce((sum, item) => sum + item.price, 0);
  }
}

export function checkout(cart) {
  return formatPrice(cart.total());
}

export const emptyCart = () => new Cart();
